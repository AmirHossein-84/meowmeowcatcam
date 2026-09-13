#!/usr/bin/env node
// Copies web assets (index.html, app.js, gestures.json, memes/) into www/
// for Capacitor `webDir`. Keeps root as source of truth.
import { cpSync, mkdirSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";

const ROOT = join(import.meta.dirname, "..");
const WWW = join(ROOT, "www");

if (existsSync(WWW)) rmSync(WWW, { recursive: true, force: true });
mkdirSync(WWW, { recursive: true });

for (const file of ["index.html", "app.js", "gestures.json"]) {
  cpSync(join(ROOT, file), join(WWW, file));
}

mkdirSync(join(WWW, "memes"), { recursive: true });
cpSync(join(ROOT, "memes"), join(WWW, "memes"), { recursive: true });

console.log("www/ prepared with index.html, app.js, gestures.json, memes/");
