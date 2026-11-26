export async function POST(request: Request) {
  const backend = process.env.BACKEND_URL || "http://127.0.0.1:8000";
  try {
    const response = await fetch(backend + "/related", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: await request.text(),
      signal: request.signal,
      cache: "no-store",
    });
    return new Response(response.body, { status: response.status, headers: {
      "content-type": response.headers.get("content-type") || "application/json",
      "cache-control": "no-store",
    } });
  } catch {
    return Response.json({ detail: "The research backend is unavailable." }, { status: 503 });
  }
}
