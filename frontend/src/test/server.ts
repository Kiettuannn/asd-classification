/**
 * MSW Node server used by Vitest (jsdom environment).
 */
import { setupServer } from 'msw/node';
import { handlers } from './handlers';

export const server = setupServer(...handlers);
