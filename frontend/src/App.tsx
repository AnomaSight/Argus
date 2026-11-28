import HowItWorksPage from "./pages/HowItWorksPage";
import InspectPage from "./pages/InspectPage";
import ResultsPage from "./pages/ResultsPage";
import { Link, usePath } from "./router";

const PAGES: Record<string, () => React.JSX.Element> = {
  "/": InspectPage,
  "/inspect": InspectPage,
  "/how": HowItWorksPage,
  "/results": ResultsPage,
};

const DOCK: [string, string][] = [
  ["/", "Inspect"],
  ["/how", "How it works"],
  ["/results", "Results"],
];

export default function App() {
  const path = usePath();
  const Page = PAGES[path] ?? InspectPage;
  const current = path === "/inspect" ? "/" : path;
  return (
    <>
      {/* Intro stage (only while html[data-intro] is set) and the big blurred backdrop word. */}
      <div className="spotlight" aria-hidden="true" />
      <div className="backdrop-word" aria-hidden="true">
        ARGUS
      </div>
      <Link to="/" className="brand">
        ARGUS
      </Link>
      <div className="app-content">
        <Page />
      </div>
      <nav className="dock" aria-label="Main">
        {DOCK.map(([to, label]) => (
          <Link key={to} to={to} aria-current={to === current ? "page" : undefined}>
            {label}
          </Link>
        ))}
      </nav>
    </>
  );
}
