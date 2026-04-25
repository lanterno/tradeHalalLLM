import { lazy, Suspense } from "react";
import { Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import { PageSkeleton } from "./components/PageSkeleton";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const Positions = lazy(() => import("./pages/Positions"));
const Trades = lazy(() => import("./pages/Trades"));
const Analytics = lazy(() => import("./pages/Analytics"));
const Sentiment = lazy(() => import("./pages/Sentiment"));
const Decisions = lazy(() => import("./pages/Decisions"));
const System = lazy(() => import("./pages/System"));
const Observability = lazy(() => import("./pages/Observability"));
const RiskAndSystem = lazy(() => import("./pages/RiskAndSystem"));

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
          path="sentiment"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <Sentiment />
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
          path="risk"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <RiskAndSystem />
            </Suspense>
          }
        />
        <Route
          path="observability"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <Observability />
            </Suspense>
          }
        />
        <Route
          path="system"
          element={
            <Suspense fallback={<PageSkeleton />}>
              <System />
            </Suspense>
          }
        />
      </Route>
    </Routes>
  );
}
