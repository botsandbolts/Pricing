"""Small first-party bridge for Streamlit's nested scrolling container.

The visible landing CTA stays an accessible semantic anchor. In Streamlit
Cloud the URL hash can change without scrolling the app's overflow container;
the bridge asks the actual in-app target to scroll into view.
No analytics, cookies, external scripts, or model dependencies.
"""
import streamlit as st

_BRIDGE = r"""
<script>
(() => {
  try {
    const host = window.parent;
    const doc = host.document;
    const key = "__cpsCalculatorAnchorHandler";
    if (host[key]) return;
    const handler = (event) => {
      const link = event.target && event.target.closest &&
                   event.target.closest('a[href="#studio-calculator"]');
      if (!link) return;
      const target = doc.getElementById("studio-calculator");
      if (!target) return; // Preserve native navigation when no target exists.
      event.preventDefault();
      const reduce = host.matchMedia && host.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const scroll = () => {
        if (!target.isConnected) return;
        target.scrollIntoView({behavior: "auto", block: "start", inline: "nearest"});
      };
      // Streamlit's main element is an overflow scroller: URL hash alone does
      // not scroll it reliably. Repeat after the click/render frame settles.
      scroll();
      host.requestAnimationFrame(scroll);
      host.setTimeout(scroll, 120);
      host.setTimeout(scroll, 420);
      try {
        host.history.replaceState(null, "", host.location.pathname + host.location.search + "#studio-calculator");
      } catch (_) { /* The price calculator is still reachable on click. */ }
      try { target.focus({preventScroll: true}); } catch (_) { /* older browsers */ }
    };
    doc.addEventListener("click", handler, true);
    host[key] = handler;
  } catch (_) {
    // Sandboxed, cross-origin hosts keep normal HTML anchor behavior.
  }
})();
</script>
"""

def install():
    """Mount once per Streamlit rerun; the parent document owns one listener."""
    st.iframe(_BRIDGE, height=0, tab_index=-1, alt="Internal calculator navigation bridge")
