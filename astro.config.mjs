import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import react from '@astrojs/react';
import mdx from '@astrojs/mdx';

export default defineConfig({
  site: 'https://shawnhdx0710.github.io',
  vite: {
    plugins: [tailwindcss()],
  },
  integrations: [react(), mdx({
    syntaxHighlight: 'shiki',
    shikiConfig: { theme: 'github-light' },
  })],
});
