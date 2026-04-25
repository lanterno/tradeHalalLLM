import { Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Positions from "./pages/Positions";
import Trades from "./pages/Trades";
import Analytics from "./pages/Analytics";
import Sentiment from "./pages/Sentiment";
import Decisions from "./pages/Decisions";
import System from "./pages/System";
import Observability from "./pages/Observability";
import RiskAndSystem from "./pages/RiskAndSystem";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="positions" element={<Positions />} />
        <Route path="trades" element={<Trades />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="sentiment" element={<Sentiment />} />
        <Route path="decisions" element={<Decisions />} />
        <Route path="risk" element={<RiskAndSystem />} />
        <Route path="observability" element={<Observability />} />
        <Route path="system" element={<System />} />
      </Route>
    </Routes>
  );
}
