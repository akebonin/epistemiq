const CACHE_NAME = 'epistemiq-v1';
const OFFLINE_URL = '/analyze'; // Or create a dedicated /offline.html

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(clients.claim());
});

self.addEventListener('fetch', (event) => {
  // Only handle GET requests
  if (event.request.method !== 'GET') return;

  event.respondWith(
    fetch(event.request)
      .catch(() => {
        // If network fails, return the offline page (or cache if you built one)
        // For now, we simply try to return the cached main page if available
        return caches.match(OFFLINE_URL);
      })
  );
});
