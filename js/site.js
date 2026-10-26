// Pluto photo list and home/personal tab switching.
const PLUTO = [
  ['first_day',            'First day in the house (9/17/2020)'],
  ['blanket',              'Laying down with his lucky blanket'],
  ['panther',              'The vicious panther stalks his unfortunate prey'],
  ['under_table',          'Hiding from his first thunderstorm'],
  ['first_sofa',           'Staging a hostile takeover of his first sofa'],
  ['business_negotiation', 'After making a shrewd business deal'],
  ['doghouse',             'Reclining in the doghouse, which he would later repurpose as a launchpad for unreachable towels'],
  ['wistful_stare',        'Staring wistfully at the pool'],
  ['second_sofa',          "As if one sofa wasn't enough, now there's two of them"],
  ['demanding',            'Trying to make us feel guilty for not giving him our food earlier'],
  ['contemplation',        'Setting his mind to the Herculean task of whether he should go downstairs or not'],
  ['side_eye',             'Pretending he did nothing wrong'],
  ['sofa',                 "Of course, you can't just take over all the sofas in one house, you must do so in another too"],
  ['after_pool',           'Refusing to stop swimming'],
  ['bed',                  'Why stop at sofas when beds are there for the taking'],
  ['refusing_dry',         'Refusing to be dried post-swim'],
  ['graduated',            'Graduated, in 20× less time than it took me', '.jpeg'],
  ['me_and_pluto',         'Pictured: a handsome young lad with a bright future. Also pictured: me.', '.jpeg']
];

function fillPluto(container, render) {
  container.innerHTML = PLUTO.map(([name, caption, ext], i) =>
    render('img/pluto/' + name + (ext || '.jpg'), caption, i)).join('');
}

function setupTabs() {
  const home = document.getElementById('home');
  const personal = document.getElementById('personal');
  function show() {
    const id = location.hash.slice(1);
    const onPersonal = id === 'personal' || (id && personal.querySelector('#' + CSS.escape(id)));
    home.hidden = !!onPersonal;
    personal.hidden = !onPersonal;
    document.querySelectorAll('[data-tab]').forEach(a =>
      a.toggleAttribute('aria-current', (a.dataset.tab === 'personal') === !!onPersonal));
    const target = id && id !== 'personal' && id !== 'home' && document.getElementById(id);
    if (target) target.scrollIntoView();
    else window.scrollTo(0, 0);
  }
  window.addEventListener('hashchange', show);
  show();
}
