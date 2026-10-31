import { NavLink, Outlet } from 'react-router-dom'

const links = [
  { to: '/customer', label: 'Lookup' },
  { to: '/campaign', label: 'Campaign' },
  { to: '/batch', label: 'Batch' },
  { to: '/segments', label: 'Segments' },
  { to: '/traffic', label: 'Traffic' },
]

export default function Layout() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <NavLink to="/" className="brand">
            NEXUS
            <span>Intelligence</span>
          </NavLink>
          <nav className="nav">
            {links.map((l) => (
              <NavLink key={l.to} to={l.to} className={({ isActive }) => (isActive ? 'active' : undefined)}>
                {l.label}
              </NavLink>
            ))}
            <NavLink
              to="/methodology"
              className={({ isActive }) => (isActive ? 'active nav-muted' : 'nav-muted')}
            >
              Methodology
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="main">
        <Outlet />
      </main>
      <footer className="footer-note">
        NEXUS · Customer behavioral intelligence · Predicted outcomes, not causal lift
      </footer>
    </div>
  )
}
