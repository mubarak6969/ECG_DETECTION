// Tiny Vercel serverless function (Node, zero dependencies) - the only
// backend logic that runs on Vercel. Its only job is to hand the static
// frontend the real inference backend's URL, which is set once as a
// Vercel Environment Variable (ECG_API_URL) rather than hardcoded, so
// the frontend never needs a build step to know where the Render
// backend lives. Never exposes anything beyond that one URL.
module.exports = (req, res) => {
    res.setHeader('Cache-Control', 'no-store');
    res.status(200).json({ apiUrl: process.env.ECG_API_URL || null });
};
