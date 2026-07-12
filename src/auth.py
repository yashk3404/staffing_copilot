"""
Item 22 -- Login / signup, backed by Supabase Auth (GoTrue).

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
bot-protection baseline for now. If real CAPTCHA is needed later
(e.g. for the public item 26 deploy), the robust way to do it is a
separate, real top-level webpage hosting Turnstile normally (not an
iframe inside Streamlit), which redirects back with a session token
in the URL -- a deliberate future build, not a bolt-on here.
"""

import os
import streamlit as st
from supabase import create_client, Client


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


def _login_form(supabase: Client) -> None:
    st.subheader("Log in")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        try:
            result = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password,
            })
            st.session_state["user"] = result.user
            st.session_state["session"] = result.session
            st.rerun()
        except Exception:
            # Deliberately generic -- Supabase's own default message
            # for bad credentials is already "Invalid login
            # credentials" and doesn't distinguish "wrong password"
            # from "no such account," which is the point (item 22's
            # exit criteria, not a bug to fix).
            st.error("Invalid login credentials.")


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
            })
            st.success("Account created. Check your email to confirm, then log in.")
        except Exception as e:
            st.error(f"Sign-up failed: {e}")


def require_login() -> None:
    """
    Call this first, before any other Streamlit UI renders. Shows
    login/signup tabs and st.stop()s the script until
    st.session_state['user'] is populated by a successful login.

    Known gap, scoped out on purpose (item 27): st.session_state does
    not survive a browser refresh, so refreshing mid-session logs the
    user out. That's a deliberate v3 cut, not an oversight here.
    """
    if st.session_state.get("user"):
        with st.sidebar:
            st.caption(f"Logged in as {st.session_state['user'].email}")
            if st.button("Log out"):
                get_supabase_client().auth.sign_out()
                for k in ("user", "session"):
                    st.session_state.pop(k, None)
                st.rerun()
        return

    supabase = get_supabase_client()
    st.title("Staffing Copilot")
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])
    with tab_login:
        _login_form(supabase)
    with tab_signup:
        _signup_form(supabase)
    st.stop()