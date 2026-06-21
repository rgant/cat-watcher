(function () {
  'use strict';

  var activeFrameIndex = 0;

  function isInputFocused() {
    var el = document.activeElement;
    return el !== null && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT');
  }

  function getFrameRows() {
    return document.querySelectorAll('.tag-row');
  }

  function setActiveFrame(index) {
    var rows = getFrameRows();
    if (rows.length === 0) {
      return;
    }
    var clamped = Math.max(0, Math.min(index, rows.length - 1));
    Array.prototype.forEach.call(rows, function (row, i) {
      if (i === clamped) {
        row.classList.add('active-frame');
      } else {
        row.classList.remove('active-frame');
      }
    });
    activeFrameIndex = clamped;
  }

  function handleKeyDown(evt) {
    if (isInputFocused()) {
      return;
    }
    var key = evt.key;

    if (key === 'ArrowUp') {
      evt.preventDefault();
      setActiveFrame(activeFrameIndex - 1);
      return;
    }

    if (key === 'ArrowDown') {
      evt.preventDefault();
      setActiveFrame(activeFrameIndex + 1);
      return;
    }

    if (key === 'ArrowRight') {
      evt.preventDefault();
      var nextLink = document.querySelector('a[rel="next"]');
      if (nextLink) {
        nextLink.click();
      }
      return;
    }

    if (key === 'ArrowLeft') {
      evt.preventDefault();
      var prevLink = document.querySelector('a[rel="prev"]');
      if (prevLink) {
        prevLink.click();
      }
      return;
    }

    if (key === 'r' || key === 'R') {
      var reviewBtn = document.getElementById('review-btn');
      if (reviewBtn) {
        reviewBtn.click();
      }
      return;
    }

    if (key === '?') {
      var overlay = document.getElementById('kbd-help');
      if (overlay && typeof overlay.showModal === 'function') {
        if (overlay.open) {
          overlay.close();
        } else {
          overlay.showModal();
        }
      }
      return;
    }

    var digit = parseInt(key, 10);
    if (!isNaN(digit) && digit >= 1 && digit <= 9) {
      var rows = getFrameRows();
      var activeRow = rows[activeFrameIndex];
      if (!activeRow) {
        return;
      }
      var buttons = activeRow.querySelectorAll('button.tag-btn');
      var target = buttons[digit - 1];
      if (target) {
        target.click();
      }
    }
  }

  document.addEventListener('keydown', handleKeyDown);

  // The dialog otherwise only opens via the '?' key; this gives it a discoverable on-screen trigger.
  var helpOpenBtn = document.getElementById('kbd-help-open');
  if (helpOpenBtn) {
    helpOpenBtn.addEventListener('click', function () {
      var overlay = document.getElementById('kbd-help');
      if (overlay && typeof overlay.showModal === 'function' && !overlay.open) {
        overlay.showModal();
      }
    });
  }
})();
