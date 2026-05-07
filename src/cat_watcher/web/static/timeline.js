// Hover preview for the timeline SVG. Renders a small floating tooltip pinned to the cursor
// when the pointer is over a per-clip <rect class="clip"> or <rect class="bucket">. No external
// deps; loaded only on the timeline page (the template includes this script tag).
(function () {
  'use strict';
  const svg = document.querySelector('.timeline-svg');
  if (svg === null) {
    return;
  }
  const tooltip = document.createElement('div');
  tooltip.className = 'timeline-tooltip';
  tooltip.setAttribute('role', 'tooltip');
  Object.assign(tooltip.style, {
    position: 'fixed',
    pointerEvents: 'none',
    display: 'none',
    background: 'oklch(20% 0.01 240)',
    color: 'oklch(95% 0 0)',
    padding: '4px 8px',
    borderRadius: '4px',
    fontSize: '12px',
    lineHeight: '1.3',
    zIndex: '1000',
    whiteSpace: 'nowrap',
  });
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

  svg.addEventListener('mouseover', function (ev) {
    const target = ev.target.closest('.clip, .bucket');
    if (target === null) {
      return;
    }
    const text = describe(target);
    if (text === null) {
      return;
    }
    tooltip.textContent = text;
    tooltip.style.display = 'block';
  });

  svg.addEventListener('mousemove', function (ev) {
    tooltip.style.left = (ev.clientX + 12) + 'px';
    tooltip.style.top = (ev.clientY + 12) + 'px';
  });

  svg.addEventListener('mouseout', function (ev) {
    if (ev.target.closest('.clip, .bucket') !== null) {
      tooltip.style.display = 'none';
    }
  });
})();
