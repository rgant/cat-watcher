/* global htmx */
/* exported handleTagResponse, handleReviewResponse */

// No auto-play after seek: operator may be triaging from a quiet room.
(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', function () {
    var video = document.querySelector('video');
    if (!video) {
      return;
    }
    var buttons = document.querySelectorAll('.contact-sheet-button');
    Array.prototype.forEach.call(buttons, function (btn) {
      btn.addEventListener('click', function () {
        var t = parseFloat(btn.dataset.seekSeconds);
        if (Number.isNaN(t)) {
          return;
        }
        video.currentTime = t;
        video.scrollIntoView({block: 'start', behavior: 'smooth'});
      });
    });
  });
})();

// Shows a transient inline error after btn, replacing any error already showing for it so rapid
// repeated failures don't stack spans.
function showInlineError(btn, status) {
  'use strict';
  var sibling = btn.nextElementSibling;
  if (sibling && sibling.classList.contains('tag-btn-error')) {
    sibling.remove();
  }
  var err = document.createElement('span');
  err.className = 'tag-btn-error';
  err.textContent = 'Save failed (' + status + ')';
  btn.insertAdjacentElement('afterend', err);
  setTimeout(function () {
    err.remove();
  }, 4000);
}

// Called by hx-on::after-request on each tag button. On 2xx, flips the button's pressed state and
// swaps hx-put/hx-delete in place. On 4xx/5xx, shows a brief inline error message next to the
// button.
function handleTagResponse(evt, btn) {
  'use strict';
  var status = evt.detail.xhr.status;
  if (status >= 200 && status < 300) {
    var pressed = btn.getAttribute('aria-pressed') === 'true';
    var url = btn.getAttribute('hx-put') || btn.getAttribute('hx-delete') || '';
    if (pressed) {
      btn.setAttribute('aria-pressed', 'false');
      btn.classList.replace('tag-btn-on', 'tag-btn-off');
      btn.removeAttribute('hx-delete');
      btn.setAttribute('hx-put', url);
    } else {
      btn.setAttribute('aria-pressed', 'true');
      btn.classList.replace('tag-btn-off', 'tag-btn-on');
      btn.removeAttribute('hx-put');
      btn.setAttribute('hx-delete', url);
    }
    // Re-process so HTMX picks up the swapped attribute.
    htmx.process(btn);
  } else {
    showInlineError(btn, status);
  }
}

// Called by hx-on::after-request on the Mark reviewed / Re-open for review button.
// On 2xx POST: flips label to "Re-open for review", swaps hx-post to hx-delete, inserts badge.
// On 2xx DELETE: flips label to "Mark reviewed", swaps hx-delete to hx-post, removes badge.
// On 4xx/5xx: shows an inline error span.
function handleReviewResponse(evt, btn) {
  'use strict';
  var status = evt.detail.xhr.status;
  if (status >= 200 && status < 300) {
    var url = btn.getAttribute('hx-post') || btn.getAttribute('hx-delete') || '';
    var wasPost = btn.hasAttribute('hx-post');
    var container = btn.parentElement;
    if (wasPost) {
      btn.removeAttribute('hx-post');
      btn.setAttribute('hx-delete', url);
      btn.textContent = 'Re-open for review';
      var today = new Date().toISOString().slice(0, 10);
      var badge = document.createElement('span');
      badge.className = 'review-badge';
      badge.textContent = 'Reviewed ' + today;
      btn.insertAdjacentElement('afterend', badge);
    } else {
      btn.removeAttribute('hx-delete');
      btn.setAttribute('hx-post', url);
      btn.textContent = 'Mark reviewed';
      if (container) {
        var existing = container.querySelector('.review-badge');
        if (existing) {
          existing.remove();
        }
      }
    }
    htmx.process(btn);
  } else {
    showInlineError(btn, status);
  }
}
