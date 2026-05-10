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
