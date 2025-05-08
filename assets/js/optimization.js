// Lazy loading for images
document.addEventListener('DOMContentLoaded', function() {
    const lazyImages = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
                observer.unobserve(img);
            }
        });
    });

    lazyImages.forEach(img => imageObserver.observe(img));
});

// Preload blog posts
const preloadPosts = () => {
    const links = document.querySelectorAll('.post-card a');
    
    const prefetchLink = (url) => {
        const link = document.createElement('link');
        link.rel = 'prefetch';
        link.href = url;
        document.head.appendChild(link);
    };

    links.forEach(link => {
        link.addEventListener('mouseover', () => {
            prefetchLink(link.href);
        });
    });
};

// Progress indicator
const initProgressBar = () => {
    const progressBar = document.createElement('div');
    progressBar.className = 'reading-progress-bar';
    document.body.appendChild(progressBar);

    window.addEventListener('scroll', () => {
        const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
        const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (winScroll / height) * 100;
        progressBar.style.width = scrolled + '%';
    });
};

// Initialize optimizations
document.addEventListener('DOMContentLoaded', () => {
    preloadPosts();
    initProgressBar();
}); 