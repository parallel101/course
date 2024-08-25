#define __builtin_ia32_ldtilecfg(__config) __asm__ volatile ("ldtilecfg\t%X0" :: "m" (*((const void **)__config)));
#define __builtin_ia32_sttilecfg(__config) __asm__ volatile ("sttilecfg\t%X0" :: "m" (*((const void **)__config)));
