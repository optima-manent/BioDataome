import { copyFile, mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const repositoryRoot = dirname(fileURLToPath(import.meta.url));
const outputDirectory = resolve(repositoryRoot, "dist/pages");

function publicationFiles(): Plugin {
  return {
    name: "publication-files",
    apply: "build",
    async closeBundle() {
      await mkdir(outputDirectory, { recursive: true });
      await copyFile(
        resolve(repositoryRoot, "app/data/atlas-graph.manifest.json"),
        resolve(outputDirectory, "release-manifest.json"),
      );
      await copyFile(
        resolve(repositoryRoot, "app/data/atlas-graph.json"),
        resolve(outputDirectory, "atlas-graph.json"),
      );
      await writeFile(resolve(outputDirectory, ".nojekyll"), "", "utf8");
    },
  };
}

export default defineConfig({
  root: resolve(repositoryRoot, "showcase"),
  base: "./",
  publicDir: resolve(repositoryRoot, "public"),
  plugins: [react(), publicationFiles()],
  build: {
    outDir: outputDirectory,
    emptyOutDir: true,
    sourcemap: false,
    target: "es2022",
  },
});
