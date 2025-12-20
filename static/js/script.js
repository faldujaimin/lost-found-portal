// static/js/script.js

// Function to preview selected image before upload
function previewImage(event) {
    var reader = new FileReader();
    reader.onload = function(){
        var output = document.getElementById('image-preview');
        output.src = reader.result;
        // Add show class to animate in
        output.classList.add('show');
        output.style.display = 'block'; // Ensure it's visible
    };
    if (event.target.files[0]) {
        reader.readAsDataURL(event.target.files[0]);
    } else {
        // If no file selected, hide the preview
        var output = document.getElementById('image-preview');
        if(output){
            output.classList.remove('show');
            // small timeout to allow transition before hiding
            setTimeout(function(){ output.style.display = 'none'; output.src = '#'; }, 300);
        }
    }
}


// Animate item cards into view using IntersectionObserver with a small stagger
document.addEventListener('DOMContentLoaded', function(){
    var grid = document.querySelector('.items-grid');
    if(!grid) return;
    grid.classList.add('animated');

    var cards = Array.prototype.slice.call(grid.querySelectorAll('.card'));
    if(cards.length === 0) return;

    var observer = new IntersectionObserver(function(entries){
        entries.forEach(function(entry){
            if(entry.isIntersecting){
                var card = entry.target;
                // add in-view class after a small delay to create a staggered effect
                var index = cards.indexOf(card);
                var delay = Math.min(200 + index * 60, 700); // cap delay
                setTimeout(function(){
                    card.classList.add('in-view');
                }, delay);
                observer.unobserve(card);
            }
        });
    }, { root: null, rootMargin: '0px', threshold: 0.12 });

    cards.forEach(function(c){ observer.observe(c); });

    // Optional: slight floating animation for images inside cards
    cards.forEach(function(c){
        var img = c.querySelector('.card-img-top');
        if(img) img.classList.add('floaty');
    });
});

// You can add more JavaScript functions here, e.g., for:
// - Client-side form validation
// - Dynamic filtering/searching without full page reload (using AJAX)
// - Interactive maps for lost/found locations (more advanced)
// - Confirmation modals for deletion

// Small utility to toggle shimmer on a loader element (example usage)
function setShimmer(selector, enable){
    var el = document.querySelector(selector);
    if(!el) return;
    if(enable) el.classList.add('shimmer'); else el.classList.remove('shimmer');
}

// Detect phone-like item names to require confirmation on Found form
document.addEventListener('DOMContentLoaded', function(){
    var itemNameInput = document.querySelector('input[name="item_name"]');
    var phoneGroup = document.getElementById('phone-confirm-group');
    var phoneConfirm = document.getElementById('phone_confirm');
    var form = document.querySelector('form[enctype="multipart/form-data"]');
    var phoneKeywords = ['phone','mobile','iphone','android','samsung','xiaomi','pixel','phone','handphone','book','bag','backpack','purse','wallet','keys','key','id','id card','card','laptop','macbook','notebook','watch','ring','jewel','jewelry','glasses','specs','spectacles','umbrella','pen','pens','pen drive','pendrive','usb','flash drive','calculator','calc','charger','cable','wire','bottle','water bottle','mouse','file','files','document','documents','chain','necklace','handsfree','hands-free','headphone','headphones','earphones','earbuds','head set','head-set'];
    // support an item box element that may be present in some templates (e.g., a selectable item list)
    var itemBox = document.getElementById('item_box');
    var confirmationBanner = document.getElementById('confirmation-banner');

    function checkForPhone(name){
        if(!name) return false;
        var lower = name.toLowerCase();
        return phoneKeywords.some(function(k){ return lower.indexOf(k) !== -1; });
    }

    function getItemName(){
        // Priority: itemBox (data-name or textContent) > input value
        if(itemBox){
            var dn = itemBox.getAttribute('data-name');
            if(dn && dn.trim()) return dn.trim();
            var txt = itemBox.textContent || itemBox.innerText || '';
            if(txt && txt.trim()) return txt.trim();
        }
        if(itemNameInput && itemNameInput.value && itemNameInput.value.trim()) return itemNameInput.value.trim();
        return '';
    }

    function updatePhoneUI(){
        var name = getItemName();
        var isPhone = checkForPhone(name || '');
        if(phoneGroup) phoneGroup.style.display = isPhone ? 'block' : 'none';

        // update label and note
        var label = document.getElementById('phone_confirm_label');
        var note = document.getElementById('phone_confirm_note');
        if(label){
            var displayName = name || 'this item';
            label.textContent = 'I confirm I put "' + displayName + '" photo like this';
        }
        if(note){
            note.textContent = 'Because this looks like ' + (name || 'a protected item') + ', we require confirmation to avoid accidental uploads.';
        }

        // update confirmation banner if present
        if(confirmationBanner){
            if(isPhone){
                confirmationBanner.style.display = 'block';
                confirmationBanner.textContent = 'Please confirm: I am uploading a photo of "' + (name || 'this item') + '".';
            } else {
                confirmationBanner.style.display = 'none';
                confirmationBanner.textContent = '';
            }
        }
    }

    if(itemNameInput){
        itemNameInput.addEventListener('input', updatePhoneUI);
    }
    // If itemBox exists, observe changes to it
    if(itemBox){
        var mo = new MutationObserver(function(){ updatePhoneUI(); });
        mo.observe(itemBox, { childList: true, subtree: true, characterData: true });
    }
    // initial run
    updatePhoneUI();

    if(form){
        form.addEventListener('submit', function(e){
            if(phoneGroup && phoneGroup.style.display !== 'none'){
                if(!phoneConfirm || !phoneConfirm.checked){
                    e.preventDefault();
                    alert('Please confirm that the uploaded image shows the phone.');
                }
            }
        });
    }
});

// Navbar toggler animation + shrink on scroll
document.addEventListener('DOMContentLoaded', function(){
    var toggler = document.querySelector('.navbar-toggler');
    var togglerIcon = document.querySelector('.toggler-icon');
    var navbar = document.querySelector('.navbar');
    if(toggler && togglerIcon){
        toggler.addEventListener('click', function(){
            togglerIcon.classList.toggle('open');
        });
    }

    // shrink on scroll
    var lastScroll = 0;
    function onScroll(){
        var cur = window.scrollY || window.pageYOffset;
        if(cur > 40){
            navbar.classList.add('shrink');
        } else {
            navbar.classList.remove('shrink');
        }
        lastScroll = cur;
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
});

// Theme toggle: add robust dark mode support, system sync, and accessibility attributes
document.addEventListener('DOMContentLoaded', function(){
    var toggle = document.getElementById('theme-toggle');
    var themeIcon = toggle ? toggle.querySelector('.theme-icon') : null;

    // Follow stored preference if present. If set to 'system' or not set, follow OS preference
    var stored = localStorage.getItem('site-theme');
    var mediaQuery = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');

    function getPreferred(){
        if(stored === 'dark' || stored === 'light') return stored;
        if(stored === 'system') return (mediaQuery && mediaQuery.matches) ? 'dark' : 'light';
        // default: follow system if available
        return (mediaQuery && mediaQuery.matches) ? 'dark' : 'light';
    }

    function updateAria(isDark){
        if(!toggle) return;
        toggle.setAttribute('aria-pressed', isDark ? 'true' : 'false');
        toggle.setAttribute('title', isDark ? 'Switch to light theme' : 'Switch to dark theme');
    }

    function applyTheme(name, persist=true){
        if(name === 'dark'){
            document.documentElement.classList.add('dark-theme');
            if(themeIcon) themeIcon.textContent = '☀️';
            updateAria(true);
        } else {
            document.documentElement.classList.remove('dark-theme');
            if(themeIcon) themeIcon.textContent = '🌙';
            updateAria(false);
        }
        // Save preference if requested (persist=false used for temporary/system-driven changes)
        if(persist){
            localStorage.setItem('site-theme', name);
            stored = name;
        }
    }

    // React to system theme changes when the user hasn't explicitly chosen (or chose 'system')
    if(mediaQuery && mediaQuery.addEventListener){
        mediaQuery.addEventListener('change', function(e){
            // respect explicit user choice (dark/light) unless they chose 'system' or had no saved value
            var s = localStorage.getItem('site-theme');
            if(!s || s === 'system'){
                var next = e.matches ? 'dark' : 'light';
                applyTheme(next, false);
            }
        });
    } else if(mediaQuery && mediaQuery.addListener){
        mediaQuery.addListener(function(e){
            var s = localStorage.getItem('site-theme');
            if(!s || s === 'system'){
                var next = e.matches ? 'dark' : 'light';
                applyTheme(next, false);
            }
        });
    }

    // initial run
    applyTheme(getPreferred(), true);

    if(toggle){
        // Standard click toggles between dark and light (explicit user choice)
        toggle.addEventListener('click', function(){
            var now = document.documentElement.classList.contains('dark-theme') ? 'light' : 'dark';
            applyTheme(now, true);
        });

        // Secondary: allow shift+click to set 'system' mode (follow OS)
        toggle.addEventListener('click', function(e){
            if(e.shiftKey){
                localStorage.setItem('site-theme', 'system');
                // apply according to current system
                applyTheme((mediaQuery && mediaQuery.matches) ? 'dark' : 'light', false);
                // brief feedback via title
                toggle.setAttribute('title', 'Following system theme (hold Shift to toggle system mode)');
            }
        });

        // ensure initial aria state
        updateAria(document.documentElement.classList.contains('dark-theme'));
    }
});