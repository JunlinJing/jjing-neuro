// Mobile device detection and style application
(function() {
    // Check if the device is mobile based on user agent or screen width
    function isMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) 
            || window.innerWidth <= 768;
    }

    // Apply mobile-specific styles and classes
    function applyMobileStyles() {
        if (isMobile()) {
            document.documentElement.classList.add('mobile');
            document.body.classList.add('mobile');
            document.documentElement.style.setProperty('--viewport-scale', '1');
        }
    }

    // Apply styles immediately
    applyMobileStyles();

    // Apply styles after DOM content loads
    document.addEventListener('DOMContentLoaded', applyMobileStyles);

    // Reapply styles on window resize
    window.addEventListener('resize', applyMobileStyles);
})(); 