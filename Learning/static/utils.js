function min(a, b) {
  if (b < a) {
    return b;
  } else {
    return a;
  }
}

function max(a, b) {
  if (b > a) {
    return b;
  } else {
    return a;
  }
}

function setVisible(selector, flag) {
  if (flag) {
    $(selector).show();
  } else {
    $(selector).hide();
  }
}