// Модель перебивания голосом: прогоняет функции bargeStep и refLevel из apps/web/static/index.html
// на синтетических уровнях микрофона. Звук описан огибающей громкости по кадрам ScriptProcessor (2048 отсчётов
// при 48 кГц), эталон — огибающая фразы ассистента по 20 мс, как её считает страница из WAV.
//
//     node tools/bargein_sim.mjs [путь к index.html] [--no-save]
//
// Проверяет: эхо ответа само его не прерывает (ложные срабатывания), голос пациента поверх эха прерывает быстро
// (время реакции от начала фразы). Эхо — копия речи ассистента с задержкой вывода звука, связью динамик→микрофон
// и разбросом ±30% (остаток эхоподавления не пропорционален). Два режима: «эталон» — голос Silero/Piper
// с огибающей WAV; «уровень» — голос браузера, эталона нет. Уровни — в единицах индикатора страницы (rms × 128):
// живая речь у микрофона ноутбука по замеру — медиана 11,7, тихая комната — 1–2.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

const pagePath = process.argv.slice(2).find((a) => !a.startsWith("--"));
const page = readFileSync(pagePath || new URL("../apps/web/static/index.html", import.meta.url), "utf8");
const code = page.split("// <barge>")[1]?.split("\n").slice(1).join("\n").split("// </barge>")[0];
if (!code) throw new Error("В index.html нет блока // <barge> … // </barge>");
const { bargeStep, refLevel } = new Function(`${code}; return { bargeStep, refLevel: typeof refLevel === "function" ? refLevel : null };`)();

const DT = (2048 / 48000) * 1000, FLOOR = 1.5, OUT = 30, SECONDS = 5, SEEDS = [7, 11, 23, 42, 99];
let seed = 7;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
// Огибающая речи: слоги ~5 в секунду, паузы между словами
const speech = (t, peak, jitter = 0.3) => {
  if (t < 0) return 0;
  const inWord = (t % 900) < 720;
  return inWord ? peak * (0.35 + 0.65 * Math.abs(Math.sin((Math.PI * t) / 200))) * (1 - jitter / 2 + jitter * rnd()) : peak * 0.08;
};

function run({ echo, user = 0, userAt = 1500, delay = 80, withRef = true }) {
  const ref20 = Array.from({ length: (SECONDS * 1000) / 20 }, (_, i) => speech(i * 20, OUT, 0.1));
  const refAt = (t) => (t < 0 ? 0 : ref20[Math.min(ref20.length - 1, Math.floor(t / 20))]);
  const s = { voiced: 0 };
  for (let t = 0; t < SECONDS * 1000; t += DT) {
    const e = (echo / OUT) * refAt(t - delay) * (0.7 + 0.6 * rnd()), u = speech(t - userAt, user);
    const rms = Math.sqrt(e * e + u * u + FLOOR * FLOOR);
    const ref = withRef && refLevel ? refLevel(ref20, t) : null;
    if (bargeStep(s, rms, DT, FLOOR, ref)) return user ? Math.round(t - userAt) : Math.round(t);
  }
  return null;
}

const rows = [];
for (const s0 of SEEDS) for (const withRef of refLevel ? [true, false] : [false]) {
  const mode = withRef ? "эталон" : "уровень", add = (r) => rows.push({ mode, seed: s0, ...r });
  seed = s0;
  for (const echo of [2, 4, 6, 8, 10]) for (const delay of [40, 150]) {
    add({ case: `только эхо ${echo}, задержка ${delay} мс`, echo, user: 0, ms: run({ echo, delay, withRef }) });
  }
  for (const echo of [2, 4, 6]) for (const user of [8, 12, 16]) {
    add({ case: `голос ${user} поверх эха ${echo}`, echo, user, ms: run({ echo, user, withRef }) });
  }
  add({ case: "голос 12 сразу с началом ответа, эхо 4", echo: 4, user: 12, ms: run({ echo: 4, user: 12, userAt: 50, withRef }) });
}

const fmt = (r) => (r.user ? (r.ms == null ? "не сработало" : `${r.ms} мс`) : r.ms == null ? "нет (верно)" : `ЛОЖНОЕ на ${r.ms} мс`);
for (const r of rows.filter((x) => x.seed === SEEDS[0])) console.log(r.mode.padEnd(8), r.case.padEnd(40), fmt(r));
const summarize = (rs) => {
  const reacts = rs.filter((r) => r.user && r.ms != null).map((r) => r.ms).sort((a, b) => a - b);
  return {
    echo_cases: rs.filter((r) => !r.user).length,
    voice_cases: rs.filter((r) => r.user).length,
    false_triggers: rs.filter((r) => !r.user && r.ms != null).map((r) => `${r.case} (сид ${r.seed})`),
    missed: rs.filter((r) => r.user && r.ms == null).map((r) => `${r.case} (сид ${r.seed})`),
    reaction_ms_median: reacts.length ? reacts[Math.floor(reacts.length / 2)] : null,
    reaction_ms_max: reacts.length ? reacts[reacts.length - 1] : null,
  };
};
const summary = Object.fromEntries([...new Set(rows.map((r) => r.mode))].map((m) => [m, summarize(rows.filter((r) => r.mode === m))]));
if (!process.argv.includes("--no-save")) {
  mkdirSync(new URL("../eval/results/", import.meta.url), { recursive: true });
  writeFileSync(new URL("../eval/results/bargein_sim.json", import.meta.url), JSON.stringify({ frame_ms: DT, rows, summary }, null, 1));
}
console.log(JSON.stringify(summary));
