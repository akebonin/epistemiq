export default async function handler(req, res) {
  try {
    const { url } = req.body;

    if (!url) {
      return res.status(400).json({ error: "Missing URL" });
    }

    // Fetch raw HTML from the article URL
    const response = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Epistemiq Article Fetcher)",
        "Accept": "text/html,application/xhtml+xml"
      }
    });

    if (!response.ok) {
      return res.status(500).json({
        error: "Failed to fetch URL",
        status: response.status
      });
    }

    const text = await response.text();

    // Return raw HTML. Frontend will clean, summarize, paste to textarea.
    res.status(200).json({ content: text });
  } catch (err) {
    console.error("fetch-article proxy error:", err);
    res.status(500).json({ error: "Internal error" });
  }
}
