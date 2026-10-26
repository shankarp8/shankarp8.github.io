// Background "sheet of paper": classical figures and equations, computed rather than drawn,
// scattered down both margins of the reading column. Colours come from the --sheet-* tokens.
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const el = (parent, tag, attrs, text) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    parent.appendChild(e);
    return e;
  };
  const pts = arr => arr.map(p => p[0].toFixed(2) + ',' + p[1].toFixed(2)).join(' ');
  const mid = (P, Q) => [(P[0] + Q[0]) / 2, (P[1] + Q[1]) / 2];
  const dist = (P, Q) => Math.hypot(P[0] - Q[0], P[1] - Q[1]);
  const foot = (P, Q, R) => { const dx = R[0] - Q[0], dy = R[1] - Q[1]; const t = ((P[0] - Q[0]) * dx + (P[1] - Q[1]) * dy) / (dx * dx + dy * dy); return [Q[0] + t * dx, Q[1] + t * dy]; };
  const meet = (p1, p2, p3, p4) => { const d = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]); const t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / d; return [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])]; };
  const line = (g, P, Q, cls) => el(g, 'line', { x1: P[0], y1: P[1], x2: Q[0], y2: Q[1], class: cls || 'ln' });
  const circle = (g, C, r, cls) => el(g, 'circle', { cx: C[0], cy: C[1], r, class: cls || 'ln' });
  const dot = (g, P, r) => el(g, 'circle', { cx: P[0], cy: P[1], r: r || 2.6, class: 'pt' });
  const label = (g, P, t, dx, dy) => el(g, 'text', { x: P[0] + (dx || 0), y: P[1] + (dy || 0), class: 'lbl' }, t);
  const path = (g, points, cls) => el(g, 'path', { d: 'M' + points.map(p => p[0].toFixed(2) + ' ' + p[1].toFixed(2)).join('L'), class: cls || 'ln' });

  function circumcenter(A, B, C) {
    const d = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]));
    const s = P => P[0] * P[0] + P[1] * P[1];
    return [(s(A) * (B[1] - C[1]) + s(B) * (C[1] - A[1]) + s(C) * (A[1] - B[1])) / d,
            (s(A) * (C[0] - B[0]) + s(B) * (A[0] - C[0]) + s(C) * (B[0] - A[0])) / d];
  }

  const FIGURES = {
    // Nine-point circle: midpoints, altitude feet, and midpoints of AH, BH, CH; N bisects OH.
    ninePoint: { box: '0 0 640 600', draw(g) {
      const A = [130, 470], B = [560, 505], C = [300, 115];
      const O = circumcenter(A, B, C), R = dist(O, A);
      const H = [A[0] + B[0] + C[0] - 2 * O[0], A[1] + B[1] + C[1] - 2 * O[1]], N = mid(O, H);
      const Fa = foot(A, B, C), Fb = foot(B, C, A), Fc = foot(C, A, B);
      circle(g, O, R, 'ln dash');
      [[A, Fa], [B, Fb], [C, Fc]].forEach(([P, F]) => line(g, P, F, 'ln dot'));
      el(g, 'polygon', { points: pts([A, B, C]), class: 'ln' });
      const u = [(H[0] - O[0]) / dist(O, H), (H[1] - O[1]) / dist(O, H)];
      line(g, [O[0] - u[0] * 260, O[1] - u[1] * 260], [H[0] + u[0] * 200, H[1] + u[1] * 200], 'ln acc dash');
      circle(g, N, R / 2, 'ln acc');
      [mid(B, C), mid(C, A), mid(A, B), Fa, Fb, Fc, mid(A, H), mid(B, H), mid(C, H)].forEach(p => dot(g, p));
      label(g, A, 'A', -18, 6); label(g, B, 'B', 8, 8); label(g, C, 'C', -4, -10);
      label(g, H, 'H', 7, -6); label(g, O, 'O', 7, 14); label(g, N, 'N', 7, -6);
    } },

    // Morley: adjacent angle trisectors meet at the corners of an equilateral triangle.
    morley: { box: '0 0 300 250', draw(g) {
      const A = [18, 222], B = [284, 222], C = [118, 22];
      const ang = (P, Q) => Math.atan2(Q[1] - P[1], Q[0] - P[0]);
      const tri = (P, Q, R) => { const a = ang(P, Q); let d = ang(P, R) - a; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return [Math.cos(a + d / 3), Math.sin(a + d / 3)]; };
      const corner = (P, Q, R) => { const u = tri(P, Q, R), v = tri(Q, P, R); return meet(P, [P[0] + u[0], P[1] + u[1]], Q, [Q[0] + v[0], Q[1] + v[1]]); };
      const M1 = corner(A, B, C), M2 = corner(B, C, A), M3 = corner(C, A, B);
      [[A, M1], [A, M3], [B, M1], [B, M2], [C, M2], [C, M3]].forEach(([P, M]) => line(g, P, M, 'ln dot'));
      el(g, 'polygon', { points: pts([A, B, C]), class: 'ln' });
      el(g, 'polygon', { points: pts([M1, M2, M3]), class: 'ln acc' });
      label(g, A, 'α', 26, -6); label(g, B, 'β', -30, -6); label(g, C, 'γ', -2, 30);
    } },

    // Apollonian gasket from Descartes' theorem (complex form).
    gasket: { box: '-1.02 -1.02 2.04 2.04', draw(g) {
      const c = (k, x, y) => ({ k, x, y, r: Math.abs(1 / k) });
      const r = 1 / (1 + 2 / Math.sqrt(3));
      const outer = c(-1, 0, 0);
      const inner = [90, 210, 330].map(d => c(1 / r, (1 - r) * Math.cos(d * Math.PI / 180), -(1 - r) * Math.sin(d * Math.PI / 180)));
      const ctr = c((1 / r) * (3 + 2 * Math.sqrt(3)), 0, 0);
      const all = [outer, ...inner, ctr];
      const refl = (a, b, e, d) => { const k = 2 * (a.k + b.k + e.k) - d.k; return c(k, (2 * (a.k * a.x + b.k * b.x + e.k * e.x) - d.k * d.x) / k, (2 * (a.k * a.y + b.k * b.y + e.k * e.y) - d.k * d.y) / k); };
      const grow = (a, b, e, d, depth) => { const n = refl(a, b, e, d); if (n.r < 0.012 || depth > 7) return; all.push(n); grow(n, b, e, a, depth + 1); grow(a, n, e, b, depth + 1); grow(a, b, n, e, depth + 1); };
      const [p, q, s] = inner;
      grow(q, s, ctr, p, 0); grow(p, s, ctr, q, 0); grow(p, q, ctr, s, 0);
      grow(outer, q, s, p, 0); grow(outer, p, s, q, 0); grow(outer, p, q, s, 0);
      all.forEach((k, i) => circle(g, [k.x, k.y], k.r, i === 4 ? 'ln acc' : 'ln'));
    } },

    // Euclid I.47: the altitude splits the square on the hypotenuse into the two leg squares.
    pythagoras: { box: '-14 6 434 466', draw(g) {
      const A = [95, 250], B = [305, 250], F = [95 + 0.36 * 210, 250];
      const h = Math.sqrt((F[0] - A[0]) * (B[0] - F[0])), C = [F[0], 250 - h];
      const sq = (P, Q, away) => { const n = [-(Q[1] - P[1]), Q[0] - P[0]]; const s = ((away[0] - P[0]) * n[0] + (away[1] - P[1]) * n[1]) > 0 ? -1 : 1; return [P, Q, [Q[0] + s * n[0], Q[1] + s * n[1]], [P[0] + s * n[0], P[1] + s * n[1]]]; };
      const s1 = sq(A, C, B), s2 = sq(C, B, A), s3 = sq(A, B, C);
      [s1, s2, s3].forEach(s => el(g, 'polygon', { points: pts(s), class: 'ln' }));
      el(g, 'polygon', { points: pts([A, B, C]), class: 'ln acc' });
      line(g, C, [F[0], s3[2][1]], 'ln dash');
      const centre = s => [(s[0][0] + s[2][0]) / 2, (s[0][1] + s[2][1]) / 2];
      label(g, centre(s1), 'b²', -10, 6); label(g, centre(s2), 'a²', -10, 6); label(g, centre(s3), 'c²', -10, 6);
    } },

    // Unit circle with the trig lines for one angle.
    unitCircle: { box: '-150 -150 300 300', draw(g) {
      const R = 110, t = 0.68, P = [R * Math.cos(t), -R * Math.sin(t)];
      line(g, [-140, 0], [140, 0]); line(g, [0, -140], [0, 140]);
      circle(g, [0, 0], R);
      line(g, [0, 0], P, 'ln acc');
      line(g, P, [P[0], 0], 'ln dash'); line(g, [R, 0], [R, -R * Math.tan(t)], 'ln dot');
      line(g, [0, 0], [R, -R * Math.tan(t)], 'ln dot');
      el(g, 'path', { d: `M 26 0 A 26 26 0 0 0 ${26 * Math.cos(t)} ${-26 * Math.sin(t)}`, class: 'ln' });
      dot(g, P);
      label(g, [30, -6], 'θ', 0, 0); label(g, P, 'sin θ', 6, 44); label(g, [P[0] / 2, 0], 'cos θ', -24, 18);
      label(g, [R, -R * Math.tan(t)], 'tan θ', 6, 30);
    } },

    // Pascal: opposite sides of a hexagon inscribed in a conic meet on one line.
    pascal: { box: '-3.4 -2.05 5.1 3.6', draw(g) {
      const a = 1.4, b = 1, P = d => [a * Math.cos(d * Math.PI / 180), -b * Math.sin(d * Math.PI / 180)];
      const p = [20, 110, 250, 150, 300, 200].map(P);
      el(g, 'ellipse', { cx: 0, cy: 0, rx: a, ry: b, class: 'ln' });
      const q = [meet(p[0], p[1], p[3], p[4]), meet(p[1], p[2], p[4], p[5]), meet(p[2], p[3], p[5], p[0])];
      // extend each hexagon side as far as its Pascal point
      [[0, 1, q[0]], [3, 4, q[0]], [1, 2, q[1]], [4, 5, q[1]], [2, 3, q[2]], [5, 0, q[2]]].forEach(([i, j, X]) => {
        const far = [p[i], p[j], X].sort((m, n) => m[0] - n[0]);
        line(g, far[0], far[2], 'ln');
      });
      const u = [q[0][0] - q[2][0], q[0][1] - q[2][1]], L = Math.hypot(u[0], u[1]);
      line(g, [q[2][0] - u[0] / L * .5, q[2][1] - u[1] / L * .5], [q[0][0] + u[0] / L * .5, q[0][1] + u[1] / L * .5], 'ln acc');
      p.forEach(v => dot(g, v, .045)); q.forEach(v => dot(g, v, .05));
    } },

    // Perpendicular bisector by compass: two equal arcs from A and B.
    compass: { box: '0 0 280 240', draw(g) {
      const A = [50, 150], B = [230, 130], r = 120, M = mid(A, B), d = dist(A, B) / 2;
      const h = Math.sqrt(r * r - d * d), n = [-(B[1] - A[1]) / (2 * d), (B[0] - A[0]) / (2 * d)];
      const X = [M[0] + n[0] * h, M[1] + n[1] * h], Y = [M[0] - n[0] * h, M[1] - n[1] * h];
      line(g, A, B); dot(g, A); dot(g, B);
      const arc = (C, P, span) => { const a0 = Math.atan2(P[1] - C[1], P[0] - C[0]); const s = [C[0] + r * Math.cos(a0 - span), C[1] + r * Math.sin(a0 - span)], e = [C[0] + r * Math.cos(a0 + span), C[1] + r * Math.sin(a0 + span)]; el(g, 'path', { d: `M ${s[0]} ${s[1]} A ${r} ${r} 0 0 1 ${e[0]} ${e[1]}`, class: 'ln' }); };
      [X, Y].forEach(P => { arc(A, P, .22); arc(B, P, .22); });
      line(g, [X[0] + n[0] * 12, X[1] + n[1] * 12], [Y[0] - n[0] * 12, Y[1] - n[1] * 12], 'ln acc');
      label(g, A, 'A', -16, 6); label(g, B, 'B', 8, 6);
    } },

    // sin x with its first Taylor polynomials.
    taylor: { box: '-170 -95 340 190', draw(g) {
      const sx = 34, sy = 52, f = (fn) => { const out = []; for (let x = -4.6; x <= 4.6; x += 0.05) { const y = fn(x); if (Math.abs(y) < 1.7) out.push([x * sx, -y * sy]); } return out; };
      line(g, [-165, 0], [165, 0]); line(g, [0, -90], [0, 90]);
      path(g, f(Math.sin), 'ln acc');
      path(g, f(x => x), 'ln dot');
      path(g, f(x => x - x ** 3 / 6), 'ln dash');
      path(g, f(x => x - x ** 3 / 6 + x ** 5 / 120), 'ln');
      label(g, [96, -60], 'sin x', 0, 0);
    } },

    // Parabola as the locus equidistant from focus and directrix; vertical rays reflect to F.
    parabola: { box: '-150 -40 300 220', draw(g) {
      const p = 30, Fc = [0, 150 - p], y = x => 150 - x * x / (4 * p);
      const curve = []; for (let x = -130; x <= 130; x += 4) if (y(x) > -30) curve.push([x, y(x)]);
      path(g, curve);
      line(g, [-140, 150 + p], [140, 150 + p], 'ln dash');
      [-90, -50, 60, 100].forEach(x => { const P = [x, y(x)]; line(g, [x, -36], P, 'ln dot'); line(g, P, Fc, 'ln acc'); });
      dot(g, Fc); label(g, Fc, 'F', 8, 4);
    } },
  };

  // Equations are written with ^{...} for superscripts and _{...} for subscripts.
  const EQUATIONS = [
    'N = ½(O + H),   R_{9} = R / 2',
    'e^{iπ} + 1 = 0',
    '(k_{1} + k_{2} + k_{3} + k_{4})^{2} = 2(k_{1}^{2} + k_{2}^{2} + k_{3}^{2} + k_{4}^{2})',
    'Σ 1/n^{2} = π^{2}/6',
    'a^{2} + b^{2} = c^{2}',
    '∫ e^{−x²} dx = √π',
    'sin^{2}θ + cos^{2}θ = 1',
    'OH^{2} = 9R^{2} − (a^{2} + b^{2} + c^{2})',
    'φ = (1 + √5) / 2',
  ];

  function equation(svg, text) {
    const t = el(svg, 'text', { x: 0, y: 24, class: 'eq' });
    let i = 0, buf = '';
    const flush = () => { if (buf) { el(t, 'tspan', {}, buf); buf = ''; } };
    while (i < text.length) {
      if ((text[i] === '^' || text[i] === '_') && text[i + 1] === '{') {
        flush();
        const end = text.indexOf('}', i);
        el(t, 'tspan', { 'baseline-shift': text[i] === '^' ? 'super' : 'sub', 'font-size': '68%' }, text.slice(i + 2, end));
        i = end + 1;
      } else buf += text[i++];
    }
    flush();
    return t;
  }

  // Where things go. top is px from the page top; side L/R is the margin it sits in.
  const LAYOUT = [
    { fig: 'ninePoint', top: 30, side: 'hero', w: 640 },
    { eq: 0, top: 590, side: 'R', w: 260, rot: -2 },
    { fig: 'morley', top: 640, side: 'L', w: 300, rot: -6 },
    { eq: 1, top: 1000, side: 'L', w: 170, rot: -4 },
    { fig: 'gasket', top: 860, side: 'R', w: 290, rot: 0 },
    { eq: 2, top: 1190, side: 'R', w: 330, rot: 3 },
    { fig: 'pythagoras', top: 1250, side: 'L', w: 270, rot: 5 },
    { eq: 4, top: 1580, side: 'L', w: 150, rot: 2 },
    { fig: 'unitCircle', top: 1480, side: 'R', w: 280, rot: -3 },
    { eq: 6, top: 1800, side: 'R', w: 200, rot: 1 },
    { fig: 'compass', top: 1880, side: 'L', w: 270, rot: -2 },
    { eq: 3, top: 2170, side: 'L', w: 160, rot: -5 },
    { fig: 'parabola', top: 2120, side: 'R', w: 280, rot: 4 },
    { fig: 'pascal', top: 2500, side: 'L', w: 330, rot: 2 },
    { eq: 7, top: 2450, side: 'R', w: 290, rot: -2 },
    { fig: 'taylor', top: 2620, side: 'R', w: 330, rot: -3 },
    { eq: 5, top: 2930, side: 'L', w: 170, rot: 3 },
    { eq: 8, top: 3040, side: 'R', w: 170, rot: -1 },
  ];
  const PERIOD = 3100; // the layout repeats below this, for long pages

  function render() {
    const sheet = document.getElementById('sheet');
    if (!sheet) return;
    sheet.textContent = '';
    const vw = document.documentElement.clientWidth;
    const col = Math.min(720, vw - 32);
    const colLeft = (vw - col) / 2;
    const height = document.documentElement.scrollHeight;
    const narrow = vw < 1100;

    const margin = colLeft - 16;
    // a short page squeezes the layout a little so the later figures still show
    const squeeze = height < PERIOD ? Math.max(height / PERIOD, 0.72) : 1;

    for (let base = 0; base < height; base += PERIOD) {
      LAYOUT.forEach(item => {
        if (base > 0 && item.side === 'hero') return;
        // equations only go where a margin can hold them, shrunk to fit if needed
        if (item.eq != null && margin < 130) return;
        const scale = narrow ? 0.8 : 1;
        let w = item.w * (item.side === 'hero' && narrow ? 0.75 : scale);
        if (item.eq != null) w = Math.min(item.w, margin - 40);
        const top = item.side === 'hero' ? item.top : base + item.top * squeeze;
        let x;
        // figures may tuck a little under the column; equations stay fully in the margin
        const tuck = item.eq != null ? -24 / w : 0.15;
        if (item.side === 'hero') x = narrow ? vw - w * 0.62 : colLeft + col + 210 - w;
        else if (item.side === 'L') x = Math.max(colLeft - w * (1 - tuck), -w * 0.4);
        else x = Math.min(colLeft + col - w * tuck, vw - w * 0.6);
        const svg = el(sheet, 'svg', { 'aria-hidden': 'true' });
        svg.style.cssText = `left:${x}px;top:${top}px;width:${w}px;` + (item.rot ? `transform:rotate(${item.rot}deg)` : '');
        if (item.fig) {
          const f = FIGURES[item.fig];
          svg.setAttribute('viewBox', f.box);
          f.draw(el(svg, 'g', {}));
        } else {
          svg.setAttribute('viewBox', `0 0 ${item.w} 34`);
          svg.setAttribute('overflow', 'visible');
          equation(svg, EQUATIONS[item.eq]);
        }
      });
    }
  }

  let raf;
  const schedule = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(render); };
  // page height changes with the tab and as photos load; the sheet itself never affects it
  new ResizeObserver(schedule).observe(document.body);
  window.addEventListener('hashchange', schedule);
  window.renderSheet = schedule;
})();
