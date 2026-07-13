"""
Item 22 -- Login / signup, backed by Supabase Auth (GoTrue).
Item 27 -- Session persistence ("remember me"), password reset, and
the email-confirmation/reset redirect target.

Import require_login() at the top of dashboard.py, before any other
UI renders. It blocks (st.stop()) until st.session_state["user"] is
set, so everything below it in dashboard.py can assume an
authenticated user without re-checking.

CAPTCHA note -- deliberately dropped, not an oversight: an earlier
version of this file rendered a Cloudflare Turnstile widget inline
via components.html() + streamlit-javascript to hand Supabase a
captcha_token. After three separate fixes (reload+query-param,
postMessage+st.html, then a split visible/invisible-iframe version
with session_state caching) it kept failing the same way: Streamlit
runs third-party JS inside a sandboxed iframe with no supported
synchronous channel back to Python, so every workaround just moved
the race condition somewhere else instead of removing it.

Rather than patch that a fourth time, CAPTCHA enforcement is turned
off in Supabase's dashboard (Authentication -> Settings -> CAPTCHA
protection) and this file sends no captcha_token. Supabase's own
built-in rate limiting on sign-up/sign-in (per IP, per email) is the
bot-protection baseline for now. If real CAPTCHA is needed later,
the robust way to do it is a separate, real top-level webpage hosting
Turnstile normally (not an iframe inside Streamlit), which redirects
back with a session token in the URL -- a deliberate future build,
not a bolt-on here.

Item 27 design notes
---------------------
**Redirect target.** sign_up() and reset_password_for_email() both
take a redirect URL to send the user back to. Left unset, Supabase
falls back to the project's dashboard-configured Site URL, which
defaults to http://localhost:3000 -- reachable from a laptop during
local dev, unreachable once this is a deployed Streamlit Cloud app.
_app_url() reads APP_URL from secrets/env (the deployed URL) so both
calls always point somewhere real. This alone fixes the "confirm my
email -> localhost refused to connect" report. It's not the only
piece, though -- see README.md's "Supabase Auth setup" section for the two dashboard settings
(Site URL, Redirect URLs allow-list) this also depends on, which
live in Supabase's dashboard and can't be set from this codebase.

**Why token_hash, not the PKCE/implicit link shapes.** Confirmation
and recovery links land the user back in this app via a URL query
param, not a fragment (#) -- Streamlit's server-rendered
st.query_params can only ever see query params; fragments never
leave the browser. That rules out both of GoTrue's other link
shapes for a server-rendered app like this one:
  - Implicit (`#access_token=...&type=recovery`) never reaches the
    server at all.
  - PKCE (`?code=...`) *is* a real query param, but exchanging it
    needs the code_verifier the client generated and stored locally
    at sign_up()/reset_password_for_email() time. get_supabase_client()
    is a single st.cache_resource singleton shared by every visitor
    on this server process, so that local storage isn't scoped per
    browser -- two people signing up at once would stomp each
    other's stored verifier, and a Streamlit Cloud restart between
    "get the email" and "click the link" (the common case) loses it
    entirely either way.
Instead, both email templates need to be set (in the Supabase
dashboard -- see README.md) to send `token_hash` + `type`, which
GoTrue verifies statelessly server-side via
`auth.verify_otp({"token_hash": ..., "type": ...})` -- no local
state to lose or collide on.

**Remember me.** st.session_state doesn't survive a browser refresh
(the actual item 22 gap this closes). On login, if "remember me" is
checked, the session's refresh_token is written to a browser cookie
via streamlit-cookies-controller. On every fresh script run with no
session_state user yet, _restore_session_from_cookie() looks for
that cookie and, if present, exchanges it for a new session --
transparent to the user, no re-entering credentials. The cookie
holds a refresh token, not the access token: refresh tokens are
long-lived and single-purpose (they can only be exchanged for a new
session, via Supabase, not used directly as a bearer credential),
which is the standard reason to prefer them for this over caching
the access token itself.

Trade-off worth being explicit about: this cookie isn't HttpOnly
(browser components can't set that from Python-side JS), so it's
readable by any JS on the page, same category of risk as the access
token Streamlit already keeps in server memory per session. Fine for
a portfolio-scale deployment; a stricter production setup would issue
its own short-lived, HttpOnly, server-signed cookie instead and use
this refresh token only server-side.
"""

import os

import streamlit as st
from supabase import create_client, Client

REMEMBER_ME_COOKIE = "sc_refresh_token"
REMEMBER_ME_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def _app_url() -> str:
    """
    Public URL this app is reachable at -- see the "Redirect target"
    note above. Set APP_URL in secrets.toml (local) or the app's
    Secrets manager (Streamlit Cloud) to the deployed URL, e.g.
    "https://staffing-copilot.streamlit.app". Falls back to the
    local dev server, where Supabase's own localhost default is
    actually the right answer.
    """
    return st.secrets.get(
        "APP_URL", os.environ.get("APP_URL", "http://localhost:8501")
    )


@st.cache_resource
def get_supabase_client() -> Client:
    """
    One client per Streamlit process. Reads from st.secrets first
    (Streamlit Cloud / local .streamlit/secrets.toml), falling back
    to env vars for parity with the GROQ_API_KEY pattern already in
    dashboard.py. Raises a clear error rather than a confusing one
    if neither is configured -- this should fail loudly at startup,
    not three functions deep into a login attempt.
    """
    url = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL"))
    key = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY"))
    if not url or not key:
        st.error(
            "Supabase isn't configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to .streamlit/secrets.toml (local) or the app's Secrets "
            "manager (Streamlit Cloud)."
        )
        st.stop()
    return create_client(url, key)


def _cookies() -> "CookieController":
    """
    Fresh instance every call is intentional and cheap -- the
    component's own __init__ already caches the browser round-trip
    in st.session_state["sc_cookies"], so this doesn't re-fetch on
    every rerun, just re-wraps the cached value.

    Import is deliberately local, not top-of-file: streamlit_cookies_
    controller imports streamlit.components.v1, which the test suite's
    lightweight fake `streamlit` module (conftest.py) doesn't provide
    -- and doesn't need to, since tests never call this function, only
    import get_supabase_client from this module.
    """
    from streamlit_cookies_controller import CookieController
    return CookieController(key="sc_cookies")


def _persist_remember_me_cookie(session) -> None:
    _cookies().set(
        REMEMBER_ME_COOKIE,
        session.refresh_token,
        max_age=REMEMBER_ME_MAX_AGE,
        same_site="lax",
    )


def _restore_session_from_cookie(supabase: Client) -> None:
    """
    "Remember me" -- silently logs the user back in from a refresh
    token stored on a previous visit, if there is one.

    Component quirk, not a bug: on the very first script run of a
    brand-new browser tab, the cookie controller hasn't heard back
    from its frontend yet and .get() returns None even when the
    cookie exists -- Streamlit auto-reruns once it does, so this
    resolves a run or two later without any extra code here. The
    visible effect is a brief flash of the login form before a
    remembered session takes over, which is an acceptable trade for
    not needing a synchronous JS bridge (see the CAPTCHA note above
    for why that path was abandoned).
    """
    refresh_token = _cookies().get(REMEMBER_ME_COOKIE)
    if not refresh_token:
        return

    try:
        result = supabase.auth.refresh_session(refresh_token)
    except Exception:
        # Expired, revoked, or garbage cookie -- clear it so this
        # doesn't retry (and fail) on every single rerun.
        _cookies().remove(REMEMBER_ME_COOKIE)
        return

    if not result.session:
        return

    st.session_state["user"] = result.user
    st.session_state["session"] = result.session
    # Refresh tokens rotate on use -- the old cookie value is now
    # dead, so re-persist the new one or the *next* visit fails.
    _persist_remember_me_cookie(result.session)


def _handle_email_link(supabase: Client) -> None:
    """
    Lands here when the user clicks a signup-confirmation or
    password-recovery email link -- per the dashboard email template
    setup in README.md, both point back at this app with
    ?token_hash=...&type=signup|recovery in the query string.

    Query params are consumed and cleared either way, so refreshing
    the page afterwards doesn't try to re-verify an already-used
    (now-invalid) token_hash.
    """
    token_hash = st.query_params.get("token_hash")
    otp_type = st.query_params.get("type")
    if not token_hash or not otp_type:
        return

    st.query_params.clear()

    try:
        result = supabase.auth.verify_otp(
            {"token_hash": token_hash, "type": otp_type}
        )
    except Exception:
        st.error(
            "That link has expired or was already used. Request a new "
            "one from the sign-up or forgot-password form below."
        )
        return

    if not result.session:
        st.error("That link didn't work. Please try again.")
        return

    st.session_state["user"] = result.user
    st.session_state["session"] = result.session

    if otp_type == "recovery":
        # Hold off on the normal dashboard until they've actually
        # set a new password -- they're authenticated at this point
        # (that's how GoTrue's recovery flow works) but "logged in
        # with the temporary state a recovery link grants" isn't the
        # same as "done resetting their password".
        st.session_state["_password_recovery"] = True
    else:
        st.success("Email confirmed! You're logged in.")


def _set_new_password_form(supabase: Client) -> None:
    st.subheader("Set a new password")
    st.caption(
        "You're in via your password-reset link. Choose a new "
        "password to finish."
    )
    with st.form("new_password_form"):
        new_password = st.text_input("New password", type="password")
        confirm = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Set password")

    if submitted:
        if not new_password or new_password != confirm:
            st.error("Passwords must match and can't be empty.")
            return
        try:
            supabase.auth.update_user({"password": new_password})
        except Exception as e:
            st.error(f"Couldn't update password: {e}")
            return
        st.session_state.pop("_password_recovery", None)
        st.success("Password updated.")
        st.rerun()


def _login_form(supabase: Client) -> None:
    st.subheader("Log in")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        remember = st.checkbox("Remember me on this device", value=True)
        submitted = st.form_submit_button("Log in")

    if submitted:
        try:
            result = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password,
            })
            st.session_state["user"] = result.user
            st.session_state["session"] = result.session
            if remember:
                _persist_remember_me_cookie(result.session)
            st.rerun()
        except Exception:
            # Deliberately generic -- Supabase's own default message
            # for bad credentials is already "Invalid login
            # credentials" and doesn't distinguish "wrong password"
            # from "no such account," which is the point (item 22's
            # exit criteria, not a bug to fix).
            st.error("Invalid login credentials.")

    with st.expander("Forgot your password?"):
        with st.form("forgot_password_form"):
            reset_email = st.text_input("Email", key="reset_email")
            reset_submitted = st.form_submit_button("Send reset link")

        if reset_submitted:
            try:
                supabase.auth.reset_password_for_email(
                    reset_email, options={"redirect_to": _app_url()},
                )
            except Exception:
                pass
            # Same message whether or not the account exists --
            # same "don't leak which emails are registered" reasoning
            # as the generic login error above.
            st.success(
                "If that email has an account, a reset link is on its way."
            )


def _signup_form(supabase: Client) -> None:
    st.subheader("Sign up")

    with st.form("signup_form"):
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        submitted = st.form_submit_button("Sign up")

    if submitted:
        try:
            supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {"email_redirect_to": _app_url()},
            })
            st.success("Account created. Check your email to confirm, then log in.")
        except Exception as e:
            st.error(f"Sign-up failed: {e}")


def require_login() -> None:
    """
    Call this first, before any other Streamlit UI renders. Shows
    login/signup tabs and st.stop()s the script until
    st.session_state['user'] is populated -- by a successful login,
    a remembered cookie, or an email-confirmation/recovery link.
    """
    supabase = get_supabase_client()

    _handle_email_link(supabase)

    if not st.session_state.get("user"):
        _restore_session_from_cookie(supabase)

    if st.session_state.get("user"):
        if st.session_state.get("_password_recovery"):
            _set_new_password_form(supabase)
            st.stop()

        with st.sidebar:
            st.caption(f"Logged in as {st.session_state['user'].email}")
            if st.button("Log out"):
                supabase.auth.sign_out()
                _cookies().remove(REMEMBER_ME_COOKIE)
                for k in ("user", "session", "_password_recovery"):
                    st.session_state.pop(k, None)
                st.rerun()
        return

    st.title("Staffing Copilot")
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])
    with tab_login:
        _login_form(supabase)
    with tab_signup:
        _signup_form(supabase)
    st.stop()