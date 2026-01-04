const CACHE_NAME = 'epistemiq-v1';
const OFFLINE_URL = '/analyze';

// 1. Define specific assets to cache immediately
// This ensures the app looks correct even when offline
const ASSETS_TO_CACHE = [
  OFFLINE_URL,                // The main page
  '/',                        // Root
  '/static/icons/logo.png',   // Your logo
  // Add Bootstrap CDN links so the UI doesn't break offline
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
  'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js'
];

// ✅ INSTALL: Cache the assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Caching all: app shell and content');
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
  self.skipWaiting();
});

// ✅ ACTIVATE: Clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(keyList.map((key) => {
        if (key !== CACHE_NAME) {
          console.log('[Service Worker] Removing old cache', key);
          return caches.delete(key);
        }
      }));
    })
  );
  return self.clients.claim();
});

// ✅ FETCH: Network First, Fallback to Cache
self.addEventListener('fetch', (event) => {
  // Only handle GET requests
  if (event.request.method !== 'GET') return;

  // Skip non-http requests (like chrome-extension://)
  if (!event.request.url.startsWith('http')) return;

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // If network works, return response
        return response;
      })
      .catch(() => {
        // ❌ Network failed.
        console.log('[Service Worker] Fetch failed; returning offline cache');

        // 1. Try to find the exact file in cache (e.g., logo.png)
        return caches.match(event.request)
          .then((cachedResponse) => {
            if (cachedResponse) {
              return cachedResponse;
            }
            
            // 2. If exact file not found, and it's a navigation (HTML) request,
            // serve the generic /analyze page so the app loads.
            if (event.request.mode === 'navigate') {
              return caches.match(OFFLINE_URL);
            }
          });
      })
  );
});
