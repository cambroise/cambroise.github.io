(() => {
  // ns-hugo:/Users/cambroise/Documents/GitHub/cambroise.github.io/themes/github.com/wowchemy/wowchemy-hugo-modules/wowchemy/assets/js/wowchemy-theming.js
  function getThemeMode() {
    return parseInt(localStorage.getItem("wcTheme") || 2);
  }
  function canChangeTheme() {
    return Boolean(window.wc.darkLightEnabled);
  }
  function initThemeVariation() {
    if (!canChangeTheme()) {
      return;
    }
    let currentThemeMode = getThemeMode();
    let isDarkTheme;
    switch (currentThemeMode) {
      case 0:
        isDarkTheme = false;
        break;
      case 1:
        isDarkTheme = true;
        break;
      default:
        if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
          isDarkTheme = true;
        } else if (window.matchMedia("(prefers-color-scheme: light)").matches) {
          isDarkTheme = false;
        } else {
          isDarkTheme = window.wc.isSiteThemeDark;
        }
        break;
    }
    if (isDarkTheme) {
      document.body.classList.add("dark");
    } else {
      document.body.classList.remove("dark");
    }
  }

  // ns-params:@params
  var wcDarkLightEnabled = true;
  var wcIsSiteThemeDark = true;

  // js/wowchemy-init.js
  window.wc = {
    darkLightEnabled: wcDarkLightEnabled,
    isSiteThemeDark: wcIsSiteThemeDark
  };
  initThemeVariation();
})();
