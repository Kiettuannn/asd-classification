/**
 * Axios HTTP client singleton for the ASD Classification backend.
 *
 * All feature-level API calls should import and use this `apiClient` instance
 * so that base URL, timeouts, and interceptors remain consistent.
 */
import axios, { type AxiosInstance, type AxiosResponse, type InternalAxiosRequestConfig } from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1';

const apiClient: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

// ---------------------------------------------------------------------------
// Request interceptor – attach auth tokens if required in future
// ---------------------------------------------------------------------------
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Placeholder: attach bearer token when auth is implemented
    return config;
  },
  (error: unknown) => Promise.reject(error),
);

// ---------------------------------------------------------------------------
// Response interceptor – normalise error shapes
// ---------------------------------------------------------------------------
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: unknown) => {
    // Re-throw to let TanStack Query handle the error state
    return Promise.reject(error);
  },
);

export default apiClient;
