import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Batch from './pages/Batch'
import Campaign from './pages/Campaign'
import Customer from './pages/Customer'
import Home from './pages/Home'
import Methodology from './pages/Methodology'
import Segments from './pages/Segments'
import Traffic from './pages/Traffic'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="customer" element={<Customer />} />
        <Route path="campaign" element={<Campaign />} />
        <Route path="batch" element={<Batch />} />
        <Route path="segments" element={<Segments />} />
        <Route path="traffic" element={<Traffic />} />
        <Route path="methodology" element={<Methodology />} />
      </Route>
    </Routes>
  )
}
