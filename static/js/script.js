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
                confirmationBanner.innerHTML = '<div class="alert alert-warning mb-0">Please confirm: I confirm I put "' + (name || 'this item') + '" photo like this</div>';
            } else {
                confirmationBanner.style.display = 'none';
                confirmationBanner.innerHTML = '';
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

// Theme toggle: add support for light/dark theme and persistence in localStorage
document.addEventListener('DOMContentLoaded', function(){
    var toggle = document.getElementById('theme-toggle');
    var themeIcon = toggle ? toggle.querySelector('.theme-icon') : null;
    var current = localStorage.getItem('site-theme') || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

    function applyTheme(name){
        if(name === 'dark'){
            document.documentElement.classList.add('dark-theme');
            if(themeIcon) themeIcon.textContent = '☀️';
        } else {
            document.documentElement.classList.remove('dark-theme');
            if(themeIcon) themeIcon.textContent = '🌙';
        }
        localStorage.setItem('site-theme', name);
    }

    // initial
    applyTheme(current);

    if(toggle){
        toggle.addEventListener('click', function(){
            var now = document.documentElement.classList.contains('dark-theme') ? 'light' : 'dark';
            applyTheme(now);
        });
    }
});