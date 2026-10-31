import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <div className="page-enter hero">
      <div>
        <p className="eyebrow">NEXUS</p>
        <h1>Know who will buy—and who to contact.</h1>
        <p>
          Score purchase likelihood, plan campaigns under a budget, and understand customer
          segments from real eCommerce clickstream behavior.
        </p>
        <div className="cta-row" style={{ marginTop: '1.4rem' }}>
          <Link className="btn btn-primary" to="/customer">
            Browse customers
          </Link>
          <Link className="btn btn-secondary" to="/campaign">
            Plan a campaign
          </Link>
        </div>
      </div>

      <div className="home-paths">
        <Link className="path-link" to="/batch">
          <h3>Score a list</h3>
          <p>Upload a CSV and download a ranked targeting file.</p>
        </Link>
        <Link className="path-link" to="/segments">
          <h3>Explore segments</h3>
          <p>Compare Champions, Explorers, and quieter shoppers.</p>
        </Link>
        <Link className="path-link" to="/traffic">
          <h3>Pick send days</h3>
          <p>See the next-week traffic outlook and launch windows.</p>
        </Link>
      </div>
    </div>
  )
}
