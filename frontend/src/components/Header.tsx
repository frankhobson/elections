import React from 'react';

interface HeaderProps {
  activeSection: string;
  scrollToSection: (id: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ activeSection, scrollToSection }) => {
  return (
    <div className="sticky-nav">
      <div className="nav-brand">🗳️ Global Election Forecaster</div>
      <div className="nav-links">
        <button
          onClick={() => scrollToSection('summary')}
          className={activeSection === 'summary' ? 'active' : ''}
        >
          Summary
        </button>
        <button
          onClick={() => scrollToSection('map-section')}
          className={activeSection === 'map-section' ? 'active' : ''}
        >
          Global Map
        </button>
        <button
          onClick={() => scrollToSection('database-section')}
          className={activeSection === 'database-section' ? 'active' : ''}
        >
          Database Explorer
        </button>
        <button
          onClick={() => scrollToSection('methodology-section')}
          className={activeSection === 'methodology-section' ? 'active' : ''}
        >
          Methodology
        </button>
        <button
          onClick={() => scrollToSection('diagnostics-section')}
          className={activeSection === 'diagnostics-section' ? 'active' : ''}
        >
          Performance
        </button>
      </div>
    </div>
  );
};
