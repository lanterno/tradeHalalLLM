import { lazy, Suspense } from "react";
import { Navigate, Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import { PageSkeleton } from "./components/PageSkeleton";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const Recommendation = lazy(() => import("./pages/Recommendation"));
const BeliefBoard = lazy(() => import("./pages/BeliefBoard"));
const Positions = lazy(() => import("./pages/Positions"));
const Trades = lazy(() => import("./pages/Trades"));
const Analytics = lazy(() => import("./pages/Analytics"));
const Decisions = lazy(() => import("./pages/Decisions"));
const Halal = lazy(() => import("./pages/Halal"));
const Core = lazy(() => import("./pages/Core"));
const Operations = lazy(() => import("./pages/Operations"));
const RiskAndSystem = lazy(() => import("./pages/RiskAndSystem"));
const Insights = lazy(() => import("./pages/Insights"));

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route
          index
          element={
            <Suspense fallback={<PageSkeleton />}>
              <Dashboard />
            </Suspense>
          }
        />
        <Route
          path="recommendation"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <Recommendation />
            </Suspense>
          }
        />
        <Route
          path="beliefs"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <BeliefBoard />
            </Suspense>
          }
        />
        <Route
          path="positions"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <Positions />
            </Suspense>
          }
        />
        <Route
          path="trades"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <Trades />
            </Suspense>
          }
        />
        <Route
          path="analytics"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <Analytics />
            </Suspense>
          }
        />
        <Route
          path="decisions"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <Decisions />
            </Suspense>
          }
        />
        <Route
          path="core"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <Core />
            </Suspense>
          }
        />
        <Route
          path="halal"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <Halal />
            </Suspense>
          }
        />
        <Route
          path="risk"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <RiskAndSystem />
            </Suspense>
          }
        />
        <Route
          path="operations"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <Operations />
            </Suspense>
          }
        />
        {/* The two pages Operations replaced. */}
        <Route path="observability" element={<Navigate to="/operations" replace />} />
        <Route path="system" element={<Navigate to="/operations" replace />} />
        <Route
          path="insights"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <Insights />
            </Suspense>
          }
        />
      </Route>
    </Routes>
  );
}
