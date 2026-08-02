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

// Re-fetches the server-computed tag_summary / has_manual_cat after a membership change and writes
// them into the Detection <dl>, so the aggregate matches a full reload without one. Recomputing in
// the browser would miss archived-subject contributions the button row doesn't render.
function updateLabelSummary(clipId) {
  'use strict';
  fetch('/clips/' + encodeURIComponent(clipId) + '/label-summary', {headers: {Accept: 'application/json'}})
    .then(function (resp) {
      return resp.ok ? resp.json() : null;
    })
    .then(function (data) {
      if (!data) {
        return;
      }
      var summaryEl = document.getElementById('detail-tag-summary');
      if (summaryEl) {
        summaryEl.textContent = data.tag_summary;
      }
      var manualEl = document.getElementById('detail-has-manual-cat');
      if (manualEl) {
        manualEl.textContent = data.has_manual_cat ? 'Yes' : 'No';
      }
    });
}

// Called by hx-on::after-request on each tag button. On 2xx, flips the button's pressed state,
// swaps hx-put/hx-delete in place, and refreshes the tag_summary. On 4xx/5xx, shows a brief inline
// error message next to the button.
function handleTagResponse(evt, btn) {
  'use strict';
  var status = evt.detail.xhr.status;
  if (status >= 200 && status < 300) {
    var pressed = btn.getAttribute('aria-pressed') === 'true';
    var url = btn.getAttribute('hx-put') || btn.getAttribute('hx-delete') || '';
    var clipId = url.split('/')[2];
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
    updateLabelSummary(clipId);
  } else {
    showInlineError(btn, status);
  }
}

// Mirrors the just-toggled reviewed state into the Detection <dl>'s reviewed_at row so it matches
// without a reload. Every string comes from the response body: the browser knows neither the
// persisted timestamp nor the configured display timezone, and a client clock near local midnight
// would render tomorrow's date.
function updateReviewedAtRow(payload) {
  'use strict';
  var dd = document.getElementById('detail-reviewed-at');
  if (!dd) {
    return;
  }
  if (payload && payload.reviewed_at_iso) {
    var time = document.createElement('time');
    time.setAttribute('datetime', payload.reviewed_at_iso);
    time.textContent = payload.reviewed_at_stamp;
    dd.replaceChildren(time);
  } else {
    dd.textContent = '—';
  }
}

// Parses the review endpoint's JSON body, or null if the response wasn't JSON.
function parseReviewPayload(xhr) {
  'use strict';
  try {
    return JSON.parse(xhr.responseText);
  } catch {
    return null;
  }
}

// Called by hx-on::after-request on the Mark reviewed / Re-open for review button.
// On 2xx POST: flips label to "Re-open for review", swaps hx-post to hx-delete, inserts badge.
// On 2xx DELETE: flips label to "Mark reviewed", swaps hx-delete to hx-post, removes badge.
// Both branches sync the Detection reviewed_at row. On 4xx/5xx: shows an inline error span.
function handleReviewResponse(evt, btn) {
  'use strict';
  var status = evt.detail.xhr.status;
  if (status >= 200 && status < 300) {
    var url = btn.getAttribute('hx-post') || btn.getAttribute('hx-delete') || '';
    var wasPost = btn.hasAttribute('hx-post');
    var container = btn.parentElement;
    var payload = parseReviewPayload(evt.detail.xhr);
    if (wasPost) {
      btn.removeAttribute('hx-post');
      btn.setAttribute('hx-delete', url);
      btn.textContent = 'Re-open for review';
      var badge = document.createElement('span');
      badge.className = 'review-badge';
      badge.textContent = 'Reviewed ' + (payload ? payload.reviewed_at_date : '');
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
    updateReviewedAtRow(wasPost ? payload : null);
    htmx.process(btn);
  } else {
    showInlineError(btn, status);
  }
}
