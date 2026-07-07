import { Component, type ErrorInfo, type ReactNode } from "react";
import { ErrorState } from "./ErrorState";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Catches render-time crashes in a routed page so one broken page shows a
 * recoverable error instead of blanking the whole app (the sidebar shell
 * stays mounted). Reset it by giving it a `key` that changes on navigation
 * (see Layout) or via the Retry button.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Surface in the console for debugging; the UI shows ErrorState.
    console.error("[dashboard] render error:", error, info.componentStack);
  }

  reset = (): void => this.setState({ error: null });

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="p-6">
          <ErrorState error={this.state.error} onRetry={this.reset} />
        </div>
      );
    }
    return this.props.children;
  }
}
