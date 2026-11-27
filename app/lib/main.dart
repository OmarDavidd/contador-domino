import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'domino.dart';

/// Aplicación: botón para subir la imagen, botón para procesarla, área de imagen y mensaje de resultado.
void main() => runApp(const DominoApp());

class DominoApp extends StatelessWidget {
  const DominoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Contador de dominó',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
      home: const CounterPage(),
    );
  }
}

/// Resultado que regresa del hilo de procesamiento (las Mat de OpenCV no pueden pasar entre hilos,
/// así que la imagen dibujada viaja como bytes PNG).
class _Summary {
  _Summary(this.count, this.items, this.doubles, this.score, this.png);
  final int count;

  /// Lo mismo que dice la imagen junto a cada ficha: "3 [1|4]", "8: 2 juntas".
  final List<String> items;
  final int doubles;
  final int score;
  final Uint8List png;
}

/// Se ejecuta en segundo plano: decodifica la foto, la analiza y codifica el resultado como PNG.
_Summary _process(Uint8List bytes) {
  final image = cv.imdecode(bytes, cv.IMREAD_COLOR);
  final r = analyze(image);
  final (_, png) = cv.imencode('.png', r.image);
  return _Summary(r.count, r.items, r.doubles, r.score, png);
}

class CounterPage extends StatefulWidget {
  const CounterPage({super.key});

  @override
  State<CounterPage> createState() => _CounterPageState();
}

class _CounterPageState extends State<CounterPage> {
  final _picker = ImagePicker();
  Uint8List? _original;
  _Summary? _result;
  bool _busy = false;
  String? _error;

  /// Carga la foto elegida (galería o cámara).
  Future<void> _pick(ImageSource source) async {
    final file = await _picker.pickImage(source: source);
    if (file == null) return;
    final bytes = await file.readAsBytes();
    setState(() {
      _original = bytes;
      _result = null;
      _error = null;
    });
  }

  Future<void> _chooseSource() async {
    final source = await showModalBottomSheet<ImageSource>(
      context: context,
      builder: (context) => SafeArea(
        child: Wrap(children: [
          ListTile(
            leading: const Icon(Icons.photo_library),
            title: const Text('Galería'),
            onTap: () => Navigator.pop(context, ImageSource.gallery),
          ),
          ListTile(
            leading: const Icon(Icons.photo_camera),
            title: const Text('Cámara'),
            onTap: () => Navigator.pop(context, ImageSource.camera),
          ),
        ]),
      ),
    );
    if (source != null) await _pick(source);
  }

  /// Botón Procesar: analiza la foto sin congelar la pantalla.
  Future<void> _processImage() async {
    final bytes = _original;
    if (bytes == null) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      // compute() recibe una función de nivel superior: solo los bytes de la foto viajan al otro hilo.
      final result = await compute(_process, bytes);
      setState(() => _result = result);
    } catch (e) {
      debugPrint('Processing failed: $e');
      setState(() => _error = 'No se pudo procesar la imagen.');
    } finally {
      setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final shown = _result?.png ?? _original;
    return Scaffold(
      appBar: AppBar(title: const Text('Contador de fichas de dominó')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _busy ? null : _chooseSource,
                    icon: const Icon(Icons.upload),
                    label: const Text('Subir imagen'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _original == null || _busy ? null : _processImage,
                    icon: const Icon(Icons.search),
                    label: const Text('Procesar'),
                  ),
                ),
              ]),
              const SizedBox(height: 16),
              Expanded(
                child: Container(
                  decoration: BoxDecoration(
                    color: theme.colorScheme.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: _busy
                      ? const Center(child: CircularProgressIndicator())
                      : shown == null
                          ? const Center(child: Text('Sube una foto de fichas de dominó'))
                          : InteractiveViewer(maxScale: 6, child: Image.memory(shown, fit: BoxFit.contain)),
                ),
              ),
              const SizedBox(height: 16),
              if (_error != null) Text(_error!, style: TextStyle(color: theme.colorScheme.error)),
              if (_result != null) _ResultPanel(result: _result!),
            ],
          ),
        ),
      ),
    );
  }
}

/// Mensaje "Se detectaron X objetos en la imagen." y el extra: valores, mulas y puntuación.
class _ResultPanel extends StatelessWidget {
  const _ResultPanel({required this.result});
  final _Summary result;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Se detectaron ${result.count} objetos en la imagen.', style: theme.textTheme.titleLarge),
        const SizedBox(height: 8),
        // Numeradas igual que en la imagen, para poder decir "la ficha 4".
        if (result.items.isNotEmpty) Text(result.items.join('    ')),
        Text('Mulas: ${result.doubles}    Puntuación total: ${result.score}'),
      ],
    );
  }
}
