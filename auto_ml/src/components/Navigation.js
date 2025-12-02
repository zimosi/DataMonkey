import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useData } from '../context/DataContext';
import './Navigation.css';

const Navigation = () => {
  const location = useLocation();
  const { jobId } = useData();

  return (
    <nav className="main-navigation">
      <Link
        to="/analysis"
        className={`nav-tab ${location.pathname === '/analysis' ? 'active' : ''} ${
          !jobId ? 'disabled' : ''
        }`}
      >
        Data Analysis
      </Link>
      <Link
        to="/automl"
        className={`nav-tab ${location.pathname === '/automl' ? 'active' : ''} ${
          !jobId ? 'disabled' : ''
        }`}
      >
        AutoML
      </Link>
    </nav>
  );
};

export default Navigation;
