import js from "@eslint/js";
import globals from "globals";
import { defineConfig } from "eslint/config";

export default defineConfig([
  {
    /*
    In your eslint.config.js file, if an ignores key is used without any other keys in the
    configuration object, then the patterns act as global ignores.
    https://eslint.org/docs/latest/use/configure/ignore#ignoring-files
    */
    ignores: [
      '.pixi',
      'tmp',
    ],
  },
  { files: ["**/*.{js,mjs,cjs}"], plugins: { js }, extends: ["js/recommended"], languageOptions: { globals: globals.browser } },
  { files: ["**/*.js"], languageOptions: { sourceType: "script" } },
]);
