import { NavLink } from 'react-router-dom';

interface NavItem {
  to: string;
  label: string;
}

const NAV_ITEMS: NavItem[] = [
  { to: '/',           label: 'Home' },
  { to: '/predict',    label: 'Predict' },
  { to: '/connectome', label: 'Connectome' },
  { to: '/cohort',     label: 'Cohort' },
];

/**
 * Top navigation bar rendered across all authenticated pages.
 * Uses React Router's NavLink for active-state styling.
 */
function Navbar() {
  return (
    <header className="sticky top-0 z-40 border-b border-gray-200 bg-white shadow-sm">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
        {/* Brand */}
        <NavLink to="/" className="flex items-center gap-2 text-brand-700 font-bold text-lg">
          <span aria-hidden="true">🧠</span>
          <span>ASD Dashboard</span>
        </NavLink>

        {/* Navigation links */}
        <nav aria-label="Main navigation">
          <ul className="flex items-center gap-1">
            {NAV_ITEMS.map(({ to, label }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={to === '/'}
                  className={({ isActive }) =>
                    [
                      'rounded-md px-3 py-2 text-sm font-medium transition-colors',
                      isActive
                        ? 'bg-brand-50 text-brand-700'
                        : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900',
                    ].join(' ')
                  }
                >
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  );
}

export { Navbar };
