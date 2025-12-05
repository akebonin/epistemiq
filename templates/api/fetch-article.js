import * as cheerio from 'cheerio';

export default async function handler(req, res) {
  try {
    const { url } = req.body;

    if (!url) {
      return res.status(400).json({ error: "Missing URL" });
    }

    // 1. Fetch raw HTML
    const response = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (compatible; EpistemiqBot/1.0; +https://epistemiq.vercel.app)",
        "Accept": "text/html,application/xhtml+xml"
      }
    });

    if (!response.ok) {
      return res.status(500).json({
        error: `Failed to fetch URL: ${response.statusText}`,
        status: response.status
      });
    }

    const rawHtml = await response.text();

    // 2. Clean HTML using Cheerio
    const $ = cheerio.load(rawHtml);

    // Remove non-content junk
    $('script, style, noscript, iframe, svg, button, header, footer, nav, aside, form, .ad, .ads, .social-share, .menu, .sidebar').remove();

    // Extract text (adds newlines between blocks to prevent word-glueing)
    // We target 'p', 'h1-h6', 'li', 'article' to get meaningful content
    let cleanText = "";
    
    // Strategy: Try to find a main article tag first
    if ($('article').length > 0) {
        cleanText = $('article').text();
    } else {
        // Fallback: Get body text
        cleanText = $('body').text();
    }

    // Normalize whitespace (collapse multiple spaces/newlines into single lines)
    cleanText = cleanText.replace(/\s\s+/g, '\n').trim();


    // 3. RUN VALIDATION (The "Landing Page / Garbage" Check)
    const validation = isValidContent(cleanText);

    if (!validation.isValid) {
        return res.status(400).json({ 
            error: `Content Rejected: ${validation.reason}. Please try pasting the text manually.` 
        });
    }

    // 4. Return Clean Text
    res.status(200).json({ content: cleanText });

  } catch (err) {
    console.error("fetch-article proxy error:", err);
    res.status(500).json({ error: "Internal server error during fetch." });
  }
}

// --- Helper: Content Quality Guard ---
function isValidContent(text) {
    if (!text || text.length < 300) {
        return { isValid: false, reason: "Text is too short (under 300 chars)" };
    }

    // A. Check for Code/CSS Garbage
    // Count braces { } and semicolons ;
    const codeChars = (text.match(/[{};]/g) || []).length;
    if (codeChars > 15 && (codeChars / text.length) > 0.02) {
        return { isValid: false, reason: "Source appears to be raw code or CSS" };
    }

    // B. Check for Menu/Landing Page Density
    // Split into lines
    const lines = text.split('\n').filter(line => line.trim().length > 0);
    
    // Count "Short Lines" (typical of menus, footers, link lists)
    // A real article has long paragraphs (>60 chars).
    const shortLineCount = lines.filter(line => line.length < 60).length;
    const ratioShort = shortLineCount / lines.length;

    // If >80% of the lines are short snippets, it's likely a homepage/nav menu
    if (lines.length > 10 && ratioShort > 0.8) {
        return { isValid: false, reason: "Source appears to be a navigation menu or landing page" };
    }

    return { isValid: true };
}
