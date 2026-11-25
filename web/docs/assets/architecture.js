// Only fetch Vue on the explorer page. The checked-in bundle also works in
// the Python-only docs environment and the dashboard's offline documentation.
if (document.getElementById('theory-explorer')) {
  const explorer = document.createElement('script')
  explorer.src = new URL('theory-explorer.js', document.currentScript.src)
  document.head.append(explorer)
}
