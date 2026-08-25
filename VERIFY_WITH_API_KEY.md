# Manual Verification Checklist — Requires GEMINI_API_KEY

Run: `export GEMINI_API_KEY="your-key" && python3 -m streamlit run app.py`

- [ ] **Success path**: Select any high-risk project → click "Generate Forensic Audit Report" → spinner appears → "Analysis Complete" banner shows → formatted report renders in expander.
- [ ] **Cache hit**: Click the same project's button a second time → report appears instantly (no spinner, no API call). Confirm via terminal: no second network request logged.
- [ ] **Invalid key**: Set a deliberately wrong key (`export GEMINI_API_KEY="bad-key"`) → click button → should show: `⚠️ Gemini API key not configured or invalid. Check your GEMINI_API_KEY environment variable and restart the app.` (NOT the "not configured" message — that only fires when key is absent entirely).
- [ ] **Rate limit (if simulatable)**: Trigger rapid repeated calls or use a quota-exhausted key → should show: `⚠️ Report generation is rate-limited right now. Please wait a few seconds and try again.`
- [ ] **Page stays usable after any error**: After any error message, confirm sidebar filters still respond and the audit table is still scrollable.
