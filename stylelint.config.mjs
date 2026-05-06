const PRECISION = 4;

/** @type {import("stylelint").Config} */
export default {
  extends: [
    'stylelint-config-standard',
    '@stylistic/stylelint-config',
    'stylelint-config-clean-order/error',
    'stylelint-plugin-defensive-css/configs/strict',
    'stylelint-plugin-logical-css/configs/recommended',
  ],
  ignoreFiles: [
    '**/*.html.jinja',
    '**/*.js',
    '**/*.json',
    '**/*.jsx',
    '**/*.md',
    '**/*.py',
    '**/*.pyc',
    '**/py.typed',
  ],
  plugins: [
    'stylelint-declaration-block-no-ignored-properties',
    'stylelint-declaration-strict-value',
    'stylelint-plugin-use-baseline',
    'stylelint-use-nesting',
  ],
  reportNeedlessDisables: true,
  rules: {
    '@stylistic/at-rule-semicolon-space-before': 'never',
    '@stylistic/block-closing-brace-newline-before': 'always',
    '@stylistic/block-closing-brace-space-after': 'always-single-line',
    '@stylistic/declaration-block-semicolon-newline-after': 'always',
    '@stylistic/declaration-block-semicolon-newline-before': 'never-multi-line',
    '@stylistic/function-comma-newline-before': 'never-multi-line',
    '@stylistic/linebreaks': 'unix',
    '@stylistic/max-line-length': 140,
    '@stylistic/media-query-list-comma-newline-before': 'never-multi-line',
    '@stylistic/named-grid-areas-alignment': true,
    '@stylistic/number-leading-zero': 'never',
    '@stylistic/selector-list-comma-newline-before': 'never-multi-line',
    '@stylistic/selector-list-comma-space-after': 'always-single-line',
    '@stylistic/string-quotes': 'single',
    '@stylistic/unicode-bom': 'never',
    '@stylistic/value-list-comma-newline-before': 'never-multi-line',
    'color-function-notation': 'modern',
    'color-named': 'never',
    'color-no-hex': true,
    'csstools/use-nesting': 'always',
    'declaration-no-important': true,
    'defensive-css/no-fixed-sizes': true, // Use the recommended properites instead of the strict list
    'defensive-css/require-at-layer': null, // BEM naming + component-scoped CSS already prevents specificity conflicts
    // Element selectors scoped under BEM classes (e.g. `.block td`) are intentional and safe;
    // the rule doesn't handle nesting
    'defensive-css/require-pure-selectors': [
      null,
      { ignoreElements: [ '*', 'html', 'body', 'code', 'pre' ] },
    ],
    // All custom properties are defined in :root (not JS-injected), so fallbacks add noise.
    // Also conflicts with declaration-strict-value requiring colors stay in variables.
    'defensive-css/require-custom-property-fallback': null,
    'function-disallowed-list': [ 'rgba', 'hsla', 'rgb', 'hsl' ],
    'max-nesting-depth': 3,
    'no-unknown-animations': true,
    'number-max-precision': [
      PRECISION,
      { insideFunctions: { '/^(oklch|oklab|lch|lab)$/': 6 } },
    ],
    'plugin/declaration-block-no-ignored-properties': true,
    'plugin/use-baseline': [
      true,
      {
        available: 'newly',
        ignoreProperties: { 'user-select': [ 'none' ] },
        severity: 'warning',
      },
    ],
    'scale-unlimited/declaration-strict-value': [
      [ '/color$/', 'z-index' ],
      {
        disableFix: true,
        ignoreValues: [ 'currentColor', 'inherit', 'transparent' ],
      },
    ],
    'selector-max-attribute': 1,
    'selector-max-class': 2,
    'selector-max-combinators': 2,
    'selector-max-compound-selectors': 3,
    'selector-max-type': 1,
    'selector-max-universal': 0,
    'selector-no-qualifying-type': [ true, { ignore: [ 'attribute' ] } ],
    'time-min-milliseconds': 100,
    'unit-disallowed-list': [ 'ch', 'cm', 'ex', 'in', 'mm', 'pc', 'pt' ],
    'value-keyword-case': [ 'lower', { camelCaseSvgKeywords: true } ],
  },
};
