/// Contador de fichas de dominó. Versión en Dart de prototipo/domino.py: mismos pasos y mismos parámetros.
///
/// Proceso:
///   1. Escala de grises + filtro mediano.
///   2. Fichas: umbral adaptativo inverso AND umbral global "muy oscuro", relleno de los puntos y apertura.
///   3. Área de una ficha: mediana de las fichas sueltas, o tilePerPip veces el área de un punto.
///   4. Manchas de n fichas: la erosión separa las que apenas se tocan; las demás se cuentan por área.
///   5. Puntos: umbral alto dentro de cada ficha; las esquinas (approxPolyDP) dan la línea entre las dos mitades.
///   6. Extra: valor de la ficha [a|b], mulas (a == b) y puntuación total.
library;

import 'dart:math' as math;

import 'package:opencv_dart/opencv_dart.dart' as cv;

// Parámetros (ajustados con las fotos de desarrollo del equipo; mismos valores que el prototipo en Python).
const int workWidth = 1000; // las imágenes se redimensionan a este ancho para comparar áreas
const int blockSize = 601; // vecindario del umbral adaptativo (impar, más grande que una ficha)
const double blockC = 25; // qué tanto más oscuro que su vecindario debe ser un píxel para ser ficha
const double pipThresh = 100; // nivel de gris a partir del cual un píxel es un punto pintado (incluye puntos gastados)
const double scalePipThresh = 140; // más estricto: solo puntos bien blancos, para medir el tamaño de un punto
const double minTileAreaRatio = 0.002; // mancha más pequeña (proporción de la imagen) que puede ser ficha
const double separateRatio = 0.18; // kernel de erosión que separa fichas que casi se tocan
const double noiseRatio = 0.06; // kernel de apertura que quita la textura del fondo
const double singleRatioMin = 1.7; // proporción largo/ancho de una ficha sola
const double singleRatioMax = 2.7;
const double thinRatio = 0.6; // un grupo de fichas mide al menos esta fracción del ancho de una ficha
const double minCore = 0.10; // parte del grupo que debe sobrevivir a esa erosión (una ficha inclinada conserva ~12 %)
const double darkThresh = 80; // las fichas son más oscuras que este nivel de gris (umbral global)
const double minDarkFraction = 0.8; // un grupo de fichas pegadas es casi todo negro...
const double blackOfBackground = 0.35; // ...y "negro" es más oscuro que el 35 % del brillo del papel en esta foto
const double minPipsPerTile = 0.5; // un grupo de n fichas tiene al menos n/2 puntos (descarta un mouse con un LED)
const int rowBand = 80; // fichas cuyos centros están a menos de esta altura forman una fila (para numerarlas)
const double tilePerPip = 110; // una ficha mide unas 110 veces el área de un punto

/// Una ficha cuyo valor se pudo leer.
class Tile {
  Tile(this.low, this.high);
  final int low;
  final int high;
  bool get isDouble => low == high;
  @override
  String toString() => '[$low|$high]';
}

/// Fichas pegadas que no se pudieron separar: se cuentan por área.
class Group {
  Group(this.count, this.pips);
  final int count;
  final int pips;
}

class DominoResult {
  DominoResult(this.tiles, this.groups, this.image, this.items);
  final List<Tile> tiles;
  final List<Group> groups;

  /// Cada ficha o grupo con el número que lleva en la imagen, en orden de lectura: "3 [1|4]", "8: 2 juntas".
  final List<String> items;

  /// Imagen BGR con las fichas dibujadas.
  final cv.Mat image;

  int get count => tiles.length + groups.fold(0, (s, g) => s + g.count);
  int get doubles => tiles.where((t) => t.isDouble).length;
  int get score => tiles.fold(0, (s, t) => s + t.low + t.high) + groups.fold(0, (s, g) => s + g.pips);
}

/// Analiza una imagen BGR y devuelve las fichas, los grupos y la imagen dibujada.
DominoResult analyze(cv.Mat bgr) {
  final scale = workWidth / bgr.cols;
  final img = cv.resize(bgr, (workWidth, (bgr.rows * scale).round()), interpolation: cv.INTER_LINEAR);
  var gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
  gray = cv.medianBlur(gray, 5);

  final mask = _tileMask(gray);
  final found = _findTiles(mask, gray);
  return _draw(img, gray, found.$1, found.$2);
}

// ---------------------------------------------------------------- utilidades

/// Kernel cuadrado de unos, de tamaño impar y mínimo 3.
cv.Mat _square(double size) {
  var s = math.max(3, size.toInt());
  if (s.isEven) s += 1;
  return cv.Mat.ones(s, s, cv.MatType.CV_8UC1);
}

cv.Mat _blank(cv.Mat like) => cv.Mat.zeros(like.rows, like.cols, cv.MatType.CV_8UC1);

cv.Mat _maskOf(cv.VecPoint contour, cv.Mat like) {
  final m = _blank(like);
  cv.drawContours(m, cv.Contours.fromVecPoint(contour), -1, cv.Scalar.all(255), thickness: -1);
  return m;
}

List<cv.VecPoint> _external(cv.Mat mask) {
  final (contours, _) = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  // Se copia cada contorno: contours[i] apunta dentro de `contours`, que el recolector de basura libera.
  return [for (var i = 0; i < contours.length; i++) cv.VecPoint.fromList(contours[i].toList())];
}

/// Dibuja rellenos los contornos externos: desaparecen los huecos de los puntos.
cv.Mat _fill(cv.Mat mask) {
  final out = _blank(mask);
  final (contours, _) = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  cv.drawContours(out, contours, -1, cv.Scalar.all(255), thickness: -1);
  return out;
}

cv.VecPoint _hull(cv.VecPoint c) => cv.VecPoint.fromList(cv.VecPoint.fromMat(cv.convexHull(c)).toList());

double _area(cv.VecPoint c) => cv.contourArea(c);

T _median<T extends num>(List<T> sorted) => sorted[sorted.length ~/ 2];

bool _touchesBorder(cv.VecPoint c, cv.Mat like) {
  final r = cv.boundingRect(c);
  return r.x <= 1 || r.y <= 1 || r.x + r.width >= like.cols - 1 || r.y + r.height >= like.rows - 1;
}

// ---------------------------------------------------------------- fichas

/// Máscara binaria con las fichas en blanco (pasos 2 y 3).
cv.Mat _tileMask(cv.Mat gray) {
  final adaptive = cv.adaptiveThreshold(
      gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, blockSize, blockC);
  // Un segundo umbral, global, quita objetos más claros que las fichas (monedas, goma, mouse gris).
  final (_, veryDark) = cv.threshold(gray, darkThresh, 255, cv.THRESH_BINARY_INV);
  var mask = cv.bitwiseAND(adaptive, veryDark);
  mask = cv.morphologyEx(mask, cv.MORPH_OPEN, _square(3));

  // Primera estimación del ancho de una ficha, solo para dimensionar el kernel de ruido.
  final minArea = minTileAreaRatio * mask.rows * mask.cols;
  final areas = _external(_fill(mask)).map(_area).where((a) => a > minArea).toList()..sort();
  if (areas.isEmpty) return mask;
  final width = math.sqrt(areas[areas.length ~/ 3] / 2);
  // PRIMERO se rellenan los huecos de los puntos y después se abre: si no, se pierden las esquinas.
  return cv.morphologyEx(_fill(mask), cv.MORPH_OPEN, _square(noiseRatio * width), iterations: 2);
}

/// Cuatro esquinas de la ficha con approxPolyDP sobre la envolvente convexa.
List<cv.Point> _corners(cv.VecPoint contour) {
  final hull = _hull(contour);
  final peri = cv.arcLength(hull, true);
  cv.VecPoint approx = hull;
  for (var eps = 0.01; eps < 0.2; eps += 0.01) {
    approx = cv.approxPolyDP(hull, eps * peri, true);
    if (approx.length <= 4) break;
  }
  if (approx.length != 4) {
    final r = cv.boundingRect(contour);
    return [
      cv.Point(r.x, r.y),
      cv.Point(r.x + r.width, r.y),
      cv.Point(r.x + r.width, r.y + r.height),
      cv.Point(r.x, r.y + r.height),
    ];
  }
  return [for (var i = 0; i < 4; i++) cv.Point(approx[i].x, approx[i].y)];
}

double _dist(cv.Point a, cv.Point b) => math.sqrt(math.pow(a.x - b.x, 2) + math.pow(a.y - b.y, 2));

/// Devuelve (solidez, proporción largo/ancho).
(double, double) _shape(cv.VecPoint contour) {
  final hullArea = _area(_hull(contour));
  final solidity = hullArea > 0 ? _area(contour) / hullArea : 0.0;
  final p = _corners(contour);
  final sides = [for (var i = 0; i < 4; i++) _dist(p[i], p[(i + 1) % 4])];
  return (solidity, sides.reduce(math.max) / math.max(sides.reduce(math.min), 1));
}

/// Criterio amplio: sólida y alargada (descarta sombras y manchas).
bool _looksLikeTile(cv.VecPoint c) {
  final (solidity, ratio) = _shape(c);
  return solidity > 0.8 && ratio > 1.4 && ratio < 3.5;
}

/// Criterio estricto: una ficha sola, sin nada pegado (sirve para medir el tamaño de una ficha).
bool _isSingle(cv.VecPoint c) {
  final (solidity, ratio) = _shape(c);
  return solidity > 0.93 && ratio > singleRatioMin && ratio < singleRatioMax;
}

/// Mediana del área de los puntos blancos (todos los puntos miden lo mismo).
double _pipArea(cv.Mat gray, cv.Mat mask) {
  final region = cv.erode(mask, _square(7));
  // Solo puntos bien blancos: los gastados son más chicos y sesgarían la medida de la ficha.
  final (_, white) = cv.threshold(gray, scalePipThresh, 255, cv.THRESH_BINARY);
  final pips = cv.morphologyEx(cv.bitwiseAND(white, region), cv.MORPH_OPEN, _square(3));
  final areas = _external(pips).map(_area).where((a) => a > 15).toList()..sort();
  if (areas.length < 3) return 0;
  // Como np.median: promedio de los dos valores centrales cuando la cantidad es par
  final n = areas.length;
  return n.isOdd ? areas[n ~/ 2] : (areas[n ~/ 2 - 1] + areas[n ~/ 2]) / 2;
}

/// Área de UNA ficha en esta foto (paso 3).
double _singleArea(List<cv.VecPoint> blobs, cv.Mat gray, cv.Mat mask) {
  final singles = blobs.where(_isSingle).map(_area).toList()..sort();
  if (singles.isNotEmpty) return _median(singles);
  final pip = _pipArea(gray, mask);
  return pip > 0 ? tilePerPip * pip : 0;
}

/// Erosiona una mancha para separar fichas que apenas se tocan y dilata cada pedazo.
List<cv.VecPoint> _split(cv.VecPoint blob, cv.Mat like, double width) {
  final single = _maskOf(blob, like);
  final kernel = _square(separateRatio * width);
  final pieces = <cv.VecPoint>[];
  for (final c in _external(cv.erode(single, kernel))) {
    if (_area(c) < 0.3 * 2 * width * width) continue;
    final piece = cv.bitwiseAND(cv.dilate(_maskOf(c, like), kernel), single); // crecer sin salirse de la mancha
    final grown = _external(piece)..sort((a, b) => _area(b).compareTo(_area(a)));
    if (grown.isNotEmpty) pieces.add(grown.first);
  }
  return pieces;
}

/// Qué parte de la mancha (sin su borde) es tan negra como una ficha.
double _darkFraction(cv.Mat gray, cv.VecPoint blob) {
  final inside = cv.erode(_maskOf(blob, gray), _square(9));
  final total = cv.countNonZero(inside);
  if (total == 0) return 0;
  // Relativo al papel (percentil 90 de la foto): así funciona con otra iluminación.
  final black = blackOfBackground * _percentile90(gray);
  // gris < black  <=>  gris <= ceil(black) - 1 (los niveles de gris son enteros)
  final (_, dark) = cv.threshold(gray, black.ceilToDouble() - 1, 255, cv.THRESH_BINARY_INV);
  return cv.countNonZero(cv.bitwiseAND(dark, inside)) / total;
}

/// Percentil 90 del brillo de la imagen (como np.percentile con interpolación lineal), a partir del histograma.
double _percentile90(cv.Mat gray) {
  final counts = List<int>.filled(256, 0);
  for (final v in gray.data) {
    counts[v]++;
  }
  final n = gray.rows * gray.cols;
  final pos = 0.9 * (n - 1);
  int valueAt(int rank) {
    var acc = 0;
    for (var v = 0; v < 256; v++) {
      acc += counts[v];
      if (acc > rank) return v;
    }
    return 255;
  }
  final lo = valueAt(pos.floor()), hi = valueAt(pos.ceil());
  return lo + (hi - lo) * (pos - pos.floor());
}

/// Un grupo de fichas es casi todo negro, tiene puntos y mide al menos lo ancho de una ficha.
bool _isTileGroup(cv.Mat gray, cv.VecPoint blob, double one, double width) {
  if (_darkFraction(gray, blob) < minDarkFraction) return false; // tarjeta, mouse
  final n = math.max(1, _roundHalfEven(_area(blob) / one));
  if (_pipCenters(gray, blob, one).length < math.max(1, minPipsPerTile * n)) return false;
  final core = cv.erode(_maskOf(blob, gray), _square(thinRatio * width)); // una pluma es más delgada que una ficha
  return cv.countNonZero(core) > minCore * _area(blob);
}

/// Separa y cuenta las fichas (paso 4). Devuelve las fichas separadas y los grupos (contorno, n).
(List<cv.VecPoint>, List<(cv.VecPoint, int)>) _findTiles(cv.Mat mask, cv.Mat gray) {
  final minArea = minTileAreaRatio * mask.rows * mask.cols;
  final blobs = _external(mask).where((b) => _area(b) > minArea && !_touchesBorder(b, mask)).toList();
  final one = _singleArea(blobs, gray, mask);
  if (one == 0) return ([], []);
  final width = math.sqrt(one / 2);

  final tiles = <cv.VecPoint>[];
  final groups = <(cv.VecPoint, int)>[];
  for (final blob in blobs) {
    final n = _roundHalfEven(_area(blob) / one);
    if (n == 0) continue; // mancha o sombra
    if (n == 1) {
      if (_looksLikeTile(blob)) tiles.add(blob);
      continue;
    }
    // Varias fichas pegadas (o una ficha tocando otro objeto oscuro).
    var rest = _maskOf(blob, gray);
    for (final piece in _split(blob, gray, width)) {
      final rel = _area(piece) / one;
      if (_isSingle(piece) && rel > 0.6 && rel < 1.4) {
        tiles.add(piece);
        cv.drawContours(rest, cv.Contours.fromVecPoint(piece), -1, cv.Scalar.all(0), thickness: -1);
        rest = cv.erode(cv.dilate(rest, _square(3)), _square(3));
      }
    }
    rest = cv.morphologyEx(rest, cv.MORPH_OPEN, _square(separateRatio * width));
    for (final left in _external(rest)) {
      final k = _roundHalfEven(_area(left) / one);
      if (k >= 1 && _isTileGroup(gray, left, one, width)) groups.add((left, k));
    }
  }
  return (tiles, groups);
}

/// Redondeo como round() de Python: los .5 van al número par.
int _roundHalfEven(double x) {
  final f = x.floorToDouble();
  final diff = x - f;
  if (diff > 0.5) return f.toInt() + 1;
  if (diff < 0.5) return f.toInt();
  return f.toInt().isEven ? f.toInt() : f.toInt() + 1;
}

// ---------------------------------------------------------------- puntos y mitades

/// Centros de los puntos blancos de una ficha (tileArea = área de UNA ficha).
List<(double, double)> _pipCenters(cv.Mat gray, cv.VecPoint tile, double tileArea) {
  var region = _maskOf(_hull(tile), gray);
  region = cv.erode(region, _square(0.16 * math.sqrt(tileArea / 2))); // ignorar el borde claro de la ficha
  final (_, white) = cv.threshold(gray, pipThresh, 255, cv.THRESH_BINARY);
  var pips = cv.bitwiseAND(white, region);
  pips = cv.morphologyEx(pips, cv.MORPH_CLOSE, _square(3)); // rellenar la pintura desgastada
  pips = cv.morphologyEx(pips, cv.MORPH_OPEN, _square(3)); // quitar píxeles claros sueltos
  final centers = <(double, double)>[];
  for (final c in _external(pips)) {
    final area = _area(c);
    final r = cv.boundingRect(c);
    final roundEnough = r.width / r.height > 0.5 && r.width / r.height < 2;
    if (area > tileArea * 0.004 && area < tileArea * 0.05 && roundEnough) {
      centers.add((r.x + r.width / 2, r.y + r.height / 2));
    }
  }
  return centers;
}

/// Dos puntos de la línea que pasa por la mitad de los lados largos.
((double, double), (double, double)) _splitLine(List<cv.Point> p) {
  (double, double) mid(cv.Point a, cv.Point b) => ((a.x + b.x) / 2, (a.y + b.y) / 2);
  if (_dist(p[0], p[1]) > _dist(p[1], p[2])) return (mid(p[0], p[1]), mid(p[2], p[3]));
  return (mid(p[1], p[2]), mid(p[3], p[0]));
}

/// Signo del producto cruz: de qué lado de la línea a-b queda el punto q.
double _side((double, double) q, (double, double) a, (double, double) b) =>
    ((b.$1 - a.$1) * (q.$2 - a.$2) - (b.$2 - a.$2) * (q.$1 - a.$1)).sign;

cv.Point _pt((double, double) p) => cv.Point(p.$1.toInt(), p.$2.toInt());

/// Orden de lectura: de arriba abajo en franjas de rowBand píxeles y, dentro de la franja, de izquierda a derecha.
(int, double) _readingOrder(cv.VecPoint c) {
  final r = cv.boundingRect(c);
  return ((r.y + r.height / 2) ~/ rowBand, r.x + r.width / 2);
}

/// Texto encima del objeto, con contorno oscuro para que se lea sobre cualquier fondo.
void _label(cv.Mat out, String text, cv.VecPoint c, cv.Scalar color) {
  final r = cv.boundingRect(c);
  final org = cv.Point(r.x, math.max(r.y - 8, 24));
  cv.putText(out, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.9, cv.Scalar(0, 0, 0), thickness: 5);
  cv.putText(out, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness: 2);
}

/// Lee el valor de cada ficha y dibuja el resultado (pasos 5 y 6). Cada ficha o grupo lleva un número.
DominoResult _draw(cv.Mat img, cv.Mat gray, List<cv.VecPoint> tiles, List<(cv.VecPoint, int)> groups) {
  final out = img.clone();
  final red = cv.Scalar(0, 0, 255);
  // n == 0 marca una ficha separada; n >= 2, un grupo de n fichas pegadas.
  final found = [for (final t in tiles) (t, 0), ...groups]..sort((x, y) {
      final (ay, ax) = _readingOrder(x.$1);
      final (by, bx) = _readingOrder(y.$1);
      return ay != by ? ay.compareTo(by) : ax.compareTo(bx);
    });
  final values = <Tile>[];
  final info = <Group>[];
  final items = <String>[];
  var number = 0;
  for (final (contour, n) in found) {
    number++;
    String text;
    cv.Scalar color;
    if (n == 0) {
      final p = _corners(contour);
      final (a, b) = _splitLine(p);
      final halves = [0, 0];
      for (final c in _pipCenters(gray, contour, _area(contour))) {
        halves[_side(c, a, b) >= 0 ? 0 : 1]++;
        cv.circle(out, _pt(c), 6, red, thickness: 2);
      }
      halves.sort();
      final t = Tile(halves[0], halves[1]);
      values.add(t);
      color = t.isDouble ? cv.Scalar(0, 200, 255) : cv.Scalar(0, 200, 0); // mulas en naranja
      cv.drawContours(out, cv.Contours.fromVecPoint(cv.VecPoint.fromList(p)), -1, color, thickness: 3);
      cv.line(out, _pt(a), _pt(b), cv.Scalar(255, 0, 0), thickness: 2);
      text = '$number $t';
    } else {
      // Fichas pegadas: se cuentan por área y sus puntos suman a la puntuación.
      final pips = _pipCenters(gray, contour, _area(contour) / n);
      for (final c in pips) {
        cv.circle(out, _pt(c), 6, red, thickness: 2);
      }
      info.add(Group(n, pips.length));
      color = cv.Scalar(255, 0, 255);
      cv.drawContours(out, cv.Contours.fromVecPoint(contour), -1, color, thickness: 3);
      text = '$number: $n juntas';
    }
    _label(out, text, contour, color);
    items.add(text);
  }
  return DominoResult(values, info, out, items);
}
