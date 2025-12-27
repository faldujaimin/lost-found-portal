const CACHE_NAME = 'lf-portal-v1';
const ASSETS_TO_CACHE = [
    '/',
    '/static/css/modern.css',
    '/static/lost_found.svg',
    '/static/js/script.js'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(ASSETS_TO_CACHE);
        })
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request).then((response) => {
            return response || fetch(event.request);
        })
    );
});
