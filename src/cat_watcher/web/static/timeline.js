// Listeners are bound on document.body via event delegation so they survive the outerHTML swap that
// range-preset clicks perform on #timeline-region.
(function () {
  'use strict';

  const tooltip = document.createElement('div');
  tooltip.className = 'timeline-tooltip';
  tooltip.setAttribute('role', 'tooltip');
  tooltip.style.display = 'none';
  document.body.appendChild(tooltip);

  function describe(target) {
    if (target.classList.contains('clip')) {
      const start = target.getAttribute('data-start') || '';
      const score = target.getAttribute('data-score') || '';
      return start + ' · score ' + score;
    }
    if (target.classList.contains('bucket')) {
      const count = target.getAttribute('data-count') || '0';
      const cat = target.getAttribute('data-cat-count') || '0';
      return count + ' clips · ' + cat + ' cat-positive';
    }
    return null;
  }

  function showTooltip(target, x, y) {
    const text = describe(target);
    if (text === null) {
      return;
    }
    tooltip.textContent = text;
    tooltip.style.left = (x + 12) + 'px';
    tooltip.style.top = (y + 12) + 'px';
    tooltip.style.display = 'block';
  }

  function hideTooltip() {
    tooltip.style.display = 'none';
  }

  // Gated on hover-capable pointers so touch devices don't get stuck-tooltip behavior.
  if (window.matchMedia('(hover: hover)').matches) {
    document.body.addEventListener('mouseover', function (ev) {
      const target = ev.target.closest('.timeline-svg .clip, .timeline-svg .bucket');
      if (target !== null) {
        showTooltip(target, ev.clientX, ev.clientY);
      }
    });

    document.body.addEventListener('mousemove', function (ev) {
      if (tooltip.style.display !== 'block') {
        return;
      }
      const target = ev.target.closest('.timeline-svg .clip, .timeline-svg .bucket');
      if (target !== null) {
        tooltip.style.left = (ev.clientX + 12) + 'px';
        tooltip.style.top = (ev.clientY + 12) + 'px';
      }
    });

    document.body.addEventListener('mouseout', function (ev) {
      if (ev.target.closest('.timeline-svg .clip, .timeline-svg .bucket') !== null) {
        hideTooltip();
      }
    });
  }

  document.body.addEventListener('focusin', function (ev) {
    const target = ev.target.closest('.timeline-svg .clip, .timeline-svg .bucket');
    if (target !== null) {
      const rect = target.getBoundingClientRect();
      showTooltip(target, rect.left + rect.width / 2, rect.bottom);
    }
  });

  document.body.addEventListener('focusout', function (ev) {
    if (ev.target.closest('.timeline-svg .clip, .timeline-svg .bucket') !== null) {
      hideTooltip();
    }
  });

  // Append to whichever #timeline-region is in the DOM at error time (HTMX swaps replace it).
  function showErrorToast() {
    const region = document.getElementById('timeline-region');
    if (region === null) {
      return;
    }
    if (region.querySelector('.timeline-error-toast') !== null) {
      return; // already present; don't stack
    }
    const toast = document.createElement('div');
    toast.className = 'timeline-error-toast';
    toast.setAttribute('role', 'alert');
    const text = document.createElement('span');
    text.textContent = 'Couldn’t load that range. Try again, or refresh.';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = 'Dismiss';
    btn.addEventListener('click', function () {
      toast.remove();
    });
    toast.appendChild(text);
    toast.appendChild(btn);
    region.insertBefore(toast, region.firstChild);
  }

  document.body.addEventListener('htmx:responseError', showErrorToast);
  document.body.addEventListener('htmx:sendError', showErrorToast);
})();
