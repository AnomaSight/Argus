// Opening sequence on the home page: a soft overhead light comes up, "ARGUS" sharpens out of
// a blur, then sinks back behind the page as a blurred backdrop while the tool rises over it.
// Driven by html[data-intro], which CSS reads; removing the attribute is the finished state.
// Plays once per browser session.
const STEPS: [phase: string, at: number][] = [
  ["lit", 120], // light comes up, the word comes into focus
  ["settle", 2000], // the word recedes into the backdrop
];
const DONE_AT = 2800;

export function startIntro() {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (location.pathname !== "/" || reduce || sessionStorage.getItem("argus-intro")) return;
  sessionStorage.setItem("argus-intro", "seen");

  const root = document.documentElement;
  root.dataset.intro = "dark";
  const timers = STEPS.map(([phase, at]) => setTimeout(() => (root.dataset.intro = phase), at));

  const finish = () => {
    timers.forEach(clearTimeout);
    clearTimeout(done);
    delete root.dataset.intro;
    window.removeEventListener("pointerdown", finish);
    window.removeEventListener("keydown", finish);
  };
  const done = setTimeout(finish, DONE_AT);
  // Any click or key skips straight to the tool.
  window.addEventListener("pointerdown", finish);
  window.addEventListener("keydown", finish);
}
