// Wires contact-sheet buttons to the <video> element so clicking a frame seeks the player to
// that frame's t_offset_seconds. Auto-play is intentionally omitted: the operator may be
// triaging from the office without sound, and a sudden audio playback would be jarring.
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
