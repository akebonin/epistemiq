// This runs on Vercel's servers, not the user's browser.
// It hides the API Key from the public.

export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { query } = req.body;

  if (!query) {
    return res.status(400).json({ error: 'Missing query' });
  }

  const CORE_API_KEY = process.env.CORE_API_KEY;

  if (!CORE_API_KEY) {
    return res.status(500).json({ error: 'Server misconfiguration: Missing API Key' });
  }

  try {
    const response = await fetch('https://api.core.ac.uk/v3/search/works', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${CORE_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        q: query,
        limit: 3
      })
    });

    if (!response.ok) {
        const errText = await response.text();
        return res.status(response.status).json({ error: `CORE API Error: ${errText}` });
    }

    const data = await response.json();
    return res.status(200).json(data);

  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}
