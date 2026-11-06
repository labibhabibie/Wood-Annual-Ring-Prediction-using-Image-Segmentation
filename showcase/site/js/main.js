/* Annual rings: section reveals, the stage that follows the reading column,
   the K-Means comparison and the draggable filter row. Everything works without it. */
(function () {
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* Ring band: each dark band grows in from the pith side */
  document.querySelectorAll('.band rect').forEach(function (r, i) { r.style.setProperty('--i', i); });

  /* Reveals, staggered inside each section */
  document.querySelectorAll('section, .card').forEach(function (s) {
    s.querySelectorAll('[data-reveal]').forEach(function (el, i) { el.style.setProperty('--rd', Math.min(i, 6) * 80 + 'ms'); });
  });
  var items = document.querySelectorAll('[data-reveal]');
  if ('IntersectionObserver' in window && !reduce) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('is-in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -8% 0px' });
    items.forEach(function (el) { io.observe(el); });
  } else {
    items.forEach(function (el) { el.classList.add('is-in'); });
  }

  /* The stage: the slice changes to the step the reader is on */
  var imgs = document.querySelectorAll('.stage__img');
  var stepName = document.querySelector('.stage__step');
  var NAMES = ['Photograph', 'Grayscale', 'K-Means, k = 6', 'Sobel, x direction'];
  var current = 0;
  function show(n) {
    if (n === current) return;
    current = n;
    imgs.forEach(function (im) { im.classList.toggle('is-on', +im.dataset.step === n); });
    if (stepName) stepName.textContent = NAMES[n];
  }
  var marks = document.querySelectorAll('[data-stage]');
  if ('IntersectionObserver' in window) {
    var so = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) show(+e.target.dataset.stage); });
    }, { rootMargin: '-45% 0px -45% 0px' });
    marks.forEach(function (m) { so.observe(m); });
  }

  /* Comparison: drag the line, pick k */
  var frame = document.querySelector('.compare__frame');
  var wipe = document.querySelector('.compare__wipe');
  if (frame && wipe) {
    var setWipe = function () { frame.style.setProperty('--wipe', wipe.value + '%'); };
    wipe.addEventListener('input', setWipe);
    setWipe();
  }
  var COLOURS = {
    2: ['#e5c4aa', '#b38154'],
    3: ['#be8959', '#7b5a39', '#e6c5aa'],
    4: ['#aa7a4d', '#c89362', '#674829', '#e6c5ab'],
    5: ['#c79260', '#a9794c', '#efcdb2', '#664728', '#d6b79d'],
    6: ['#b4804f', '#8e6a48', '#5c3e1f', '#efcdb2', '#ca9564', '#d5b69d'],
    7: ['#8c6947', '#ae9179', '#b57f4d', '#cd9460', '#d7b79e', '#efcdb2', '#5b3d1e'],
    8: ['#d7b89f', '#593b1d', '#af927a', '#ac7848', '#856748', '#d29863', '#bf8855', '#f0cdb2']
  };
  var WORDS = { 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight' };
  var sw = document.querySelector('.kread__swatches');
  var kEl = document.querySelector('.kread__k');
  var ageEl = document.querySelector('.kread__age');
  function lum(hex) {
    var n = parseInt(hex.slice(1), 16);
    return 0.299 * (n >> 16) + 0.587 * (n >> 8 & 255) + 0.114 * (n & 255);
  }
  function setK(k) {
    var cs = COLOURS[k].slice().sort(function (a, b) { return lum(a) - lum(b); });
    sw.innerHTML = cs.map(function (c, i) { return '<b style="--c:' + c + ';--i:' + i + '"></b>'; }).join('');
    kEl.textContent = WORDS[k] + ' colours.';
    ageEl.textContent = k + ' years old';
  }
  document.querySelectorAll('.kpick input').forEach(function (r) {
    r.addEventListener('change', function () { if (r.checked) setK(+r.value); });
  });
  if (sw) setK(+(document.querySelector('.kpick input:checked') || { value: 6 }).value);

  /* Filter row: drag sideways with a mouse */
  var row = document.querySelector('.row');
  if (row) {
    var down = false, startX = 0, startLeft = 0, moved = false;
    row.addEventListener('pointerdown', function (e) {
      if (e.pointerType !== 'mouse') return;
      down = true; moved = false; startX = e.clientX; startLeft = row.scrollLeft;
    });
    window.addEventListener('pointermove', function (e) {
      if (!down) return;
      var dx = e.clientX - startX;
      if (Math.abs(dx) > 4) { moved = true; row.classList.add('is-drag'); }
      row.scrollLeft = startLeft - dx;
    });
    window.addEventListener('pointerup', function () { down = false; row.classList.remove('is-drag'); });
    row.addEventListener('click', function (e) { if (moved) { e.preventDefault(); e.stopPropagation(); } }, true);
  }
})();
