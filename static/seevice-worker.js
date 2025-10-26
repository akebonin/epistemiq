// service-worker.js

// This is a placeholder service worker.
// For full PWA offline capabilities, you would add caching strategies here.
// For now, it mainly enables PWA installability and share_target functionality.

self.addEventListener('install', (event) => {
  console.log('Service Worker: Installed');
  self.skipWaiting(); // Forces the waiting service worker to become the active service worker.
});

self.addEventListener('activate', (event) => {
  console.log('Service Worker: Activated');
  event.waitUntil(clients.claim()); // Takes control of clients (pages) within its scope immediately.
});

// Basic fetch handler (you can expand this for caching assets)
self.addEventListener('fetch', (event) => {
  // console.log('Service Worker: Fetching', event.request.url);
  // For now, we just let all network requests pass through.
  // To enable offline capabilities, you'd add caching logic here.
  event.respondWith(fetch(event.request));
});

// You might also add sync or push notification handlers here if needed in the future.
