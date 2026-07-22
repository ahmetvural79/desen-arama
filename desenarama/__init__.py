"""Desen Arama — Halı deseni görsel benzerlik arama uygulaması.

Çekirdek motor (``desenarama.core``) GUI'den tamamen bağımsızdır ve
komut satırından ya da PySide6 arayüzünden kullanılabilir. Uygulama
Windows üzerinde, yerel veya ağ (SMB/UNC) klasörlerinde çalışacak
şekilde tasarlanmıştır; tüm türetilmiş veriler (indeks, thumbnail,
vektörler) daima yerel disktedir.
"""

__version__ = "1.0.0"
__app_name__ = "Desen Arama"
__app_id__ = "DesenArama"  # %LOCALAPPDATA%\DesenArama

__all__ = ["__version__", "__app_name__", "__app_id__"]
