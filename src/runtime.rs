/// Bruto runtime — shared LLVM IR functions for memory, I/O, and capture.
///
/// Call `emit_runtime(context, module)` to inject all `bruto_*` functions
/// into an LLVM module.  Each is a thin wrapper around libc.

use inkwell::AddressSpace;
use inkwell::context::Context;
use inkwell::module::Module;

/// Emit all bruto runtime function definitions into `module`.
///
/// After this call the module contains callable functions:
///   bruto_alloc, bruto_free,
///   bruto_write_int, bruto_write_str, bruto_write_bool, bruto_writeln,
///   bruto_read_int,
///   bruto_capture_open, bruto_capture_write_int, bruto_capture_write_str,
///   bruto_capture_write_bool, bruto_capture_writeln, bruto_capture_close
pub fn emit_runtime<'ctx>(context: &'ctx Context, module: &Module<'ctx>) {
    let i32_ty = context.i32_type();
    let i64_ty = context.i64_type();
    let i1_ty = context.bool_type();
    let void_ty = context.void_type();
    let ptr_ty = context.ptr_type(AddressSpace::default());

    // ── libc externs ────────────────────────────────────────
    let printf_ty = i32_ty.fn_type(&[ptr_ty.into()], true);
    module.add_function("printf", printf_ty, None);

    let putchar_ty = i32_ty.fn_type(&[i32_ty.into()], false);
    module.add_function("putchar", putchar_ty, None);

    let scanf_ty = i32_ty.fn_type(&[ptr_ty.into()], true);
    module.add_function("scanf", scanf_ty, None);

    let malloc_ty = ptr_ty.fn_type(&[i64_ty.into()], false);
    module.add_function("malloc", malloc_ty, None);

    let free_ty = void_ty.fn_type(&[ptr_ty.into()], false);
    module.add_function("free", free_ty, None);

    let fopen_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
    module.add_function("fopen", fopen_ty, None);

    let fclose_ty = i32_ty.fn_type(&[ptr_ty.into()], false);
    module.add_function("fclose", fclose_ty, None);

    let fprintf_ty = i32_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], true);
    module.add_function("fprintf", fprintf_ty, None);

    let fflush_ty = i32_ty.fn_type(&[ptr_ty.into()], false);
    module.add_function("fflush", fflush_ty, None);

    // ── global: capture FILE* ───────────────────────────────
    let capture_global = module.add_global(ptr_ty, None, "_bruto_capture_fp");
    capture_global.set_initializer(&ptr_ty.const_null());

    // ── bruto_alloc(i64) -> ptr ─────────────────────────────
    {
        let fn_ty = ptr_ty.fn_type(&[i64_ty.into()], false);
        let func = module.add_function("bruto_alloc", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let size = func.get_first_param().unwrap().into_int_value();
        let malloc = module.get_function("malloc").unwrap();
        let ptr = b.build_call(malloc, &[size.into()], "ptr").unwrap()
            .try_as_basic_value().basic().unwrap();
        b.build_return(Some(&ptr)).unwrap();
    }

    // ── bruto_free(ptr) ─────────────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_free", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let p = func.get_first_param().unwrap().into_pointer_value();
        let free = module.get_function("free").unwrap();
        b.build_call(free, &[p.into()], "").unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_write_int(i64) ────────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[i64_ty.into()], false);
        let func = module.add_function("bruto_write_int", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let val = func.get_first_param().unwrap().into_int_value();
        let printf = module.get_function("printf").unwrap();
        let fmt = b.build_global_string_ptr("%lld", "fmt_int").unwrap();
        b.build_call(printf, &[fmt.as_pointer_value().into(), val.into()], "").unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_write_str(ptr) ────────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_write_str", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let printf = module.get_function("printf").unwrap();
        let fmt = b.build_global_string_ptr("%s", "fmt_str").unwrap();
        b.build_call(printf, &[fmt.as_pointer_value().into(), s.into()], "").unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_write_bool(i1) ────────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[i1_ty.into()], false);
        let func = module.add_function("bruto_write_bool", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let val = func.get_first_param().unwrap().into_int_value();
        let true_s = b.build_global_string_ptr("true", "true_s").unwrap();
        let false_s = b.build_global_string_ptr("false", "false_s").unwrap();
        let sel = b.build_select(val, true_s.as_pointer_value(), false_s.as_pointer_value(), "bs").unwrap();
        let printf = module.get_function("printf").unwrap();
        let fmt = b.build_global_string_ptr("%s", "fmt_bs").unwrap();
        b.build_call(printf, &[fmt.as_pointer_value().into(), sel.into()], "").unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_write_real(f64) ──────────────────────────────
    {
        let f64_ty = context.f64_type();
        let fn_ty = void_ty.fn_type(&[f64_ty.into()], false);
        let func = module.add_function("bruto_write_real", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let val = func.get_first_param().unwrap().into_float_value();
        let printf = module.get_function("printf").unwrap();
        let fmt = b.build_global_string_ptr("%.10g", "fmt_real").unwrap();
        b.build_call(printf, &[fmt.as_pointer_value().into(), val.into()], "").unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_write_char(i8) ────────────────────────────────
    {
        let i8_ty = context.i8_type();
        let fn_ty = void_ty.fn_type(&[i8_ty.into()], false);
        let func = module.add_function("bruto_write_char", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let val = func.get_first_param().unwrap().into_int_value();
        let putchar = module.get_function("putchar").unwrap();
        let ext = b.build_int_z_extend(val, i32_ty, "zext").unwrap();
        b.build_call(putchar, &[ext.into()], "").unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_writeln() ─────────────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[], false);
        let func = module.add_function("bruto_writeln", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let putchar = module.get_function("putchar").unwrap();
        b.build_call(putchar, &[i32_ty.const_int(10, false).into()], "").unwrap();
        b.build_return(None).unwrap();
    }

    // ── bruto_read_int() -> i64 ─────────────────────────────
    {
        let fn_ty = i64_ty.fn_type(&[], false);
        let func = module.add_function("bruto_read_int", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let tmp = b.build_alloca(i64_ty, "tmp").unwrap();
        b.build_store(tmp, i64_ty.const_int(0, false)).unwrap();
        let scanf = module.get_function("scanf").unwrap();
        let fmt = b.build_global_string_ptr("%lld", "fmt_scan").unwrap();
        b.build_call(scanf, &[fmt.as_pointer_value().into(), tmp.into()], "").unwrap();
        let val = b.build_load(i64_ty, tmp, "val").unwrap();
        b.build_return(Some(&val)).unwrap();
    }

    // ── bruto_capture_open(ptr path) ────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_capture_open", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let path = func.get_first_param().unwrap().into_pointer_value();
        let fopen = module.get_function("fopen").unwrap();
        let mode = b.build_global_string_ptr("w", "cap_mode").unwrap();
        let fp = b.build_call(fopen, &[path.into(), mode.as_pointer_value().into()], "fp").unwrap()
            .try_as_basic_value().basic().unwrap();
        let g = module.get_global("_bruto_capture_fp").unwrap();
        b.build_store(g.as_pointer_value(), fp).unwrap();
        b.build_return(None).unwrap();
    }

    // helper: load capture fp, return (builder, fp_val, is_null_bb, write_bb)
    // We'll inline the pattern for each capture_write variant instead.

    // ── bruto_capture_write_int(i64) ────────────────────────
    emit_capture_write_fn(context, module, "bruto_capture_write_int",
        &[i64_ty.into()], "%lld");

    // ── bruto_capture_write_str(ptr) ────────────────────────
    emit_capture_write_fn(context, module, "bruto_capture_write_str",
        &[ptr_ty.into()], "%s");

    // ── bruto_capture_write_bool(i1) — needs special handling
    {
        let fn_ty = void_ty.fn_type(&[i1_ty.into()], false);
        let func = module.add_function("bruto_capture_write_bool", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);

        let g = module.get_global("_bruto_capture_fp").unwrap();
        let fp = b.build_load(ptr_ty, g.as_pointer_value(), "fp").unwrap().into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();

        let write_bb = context.append_basic_block(func, "write");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, end_bb, write_bb).unwrap();

        b.position_at_end(write_bb);
        let val = func.get_first_param().unwrap().into_int_value();
        let true_s = b.build_global_string_ptr("true", "cap_true").unwrap();
        let false_s = b.build_global_string_ptr("false", "cap_false").unwrap();
        let sel = b.build_select(val, true_s.as_pointer_value(), false_s.as_pointer_value(), "bs").unwrap();
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b.build_global_string_ptr("%s", "cap_fmt_bs").unwrap();
        b.build_call(fprintf, &[fp.into(), fmt.as_pointer_value().into(), sel.into()], "").unwrap();
        let fflush = module.get_function("fflush").unwrap();
        b.build_call(fflush, &[fp.into()], "").unwrap();
        b.build_unconditional_branch(end_bb).unwrap();

        b.position_at_end(end_bb);
        b.build_return(None).unwrap();
    }

    // ── bruto_capture_write_real(f64) ─────────────────────
    emit_capture_write_fn(context, module, "bruto_capture_write_real",
        &[context.f64_type().into()], "%.10g");

    // ── bruto_capture_write_char(i8) ────────────────────
    {
        let i8_ty = context.i8_type();
        let fn_ty = void_ty.fn_type(&[i8_ty.into()], false);
        let func = module.add_function("bruto_capture_write_char", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);

        let g = module.get_global("_bruto_capture_fp").unwrap();
        let fp = b.build_load(ptr_ty, g.as_pointer_value(), "fp").unwrap().into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();

        let write_bb = context.append_basic_block(func, "write");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, end_bb, write_bb).unwrap();

        b.position_at_end(write_bb);
        let val = func.get_first_param().unwrap().into_int_value();
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b.build_global_string_ptr("%c", "cap_fmt_ch").unwrap();
        let ext = b.build_int_z_extend(val, i32_ty, "zext").unwrap();
        b.build_call(fprintf, &[fp.into(), fmt.as_pointer_value().into(), ext.into()], "").unwrap();
        let fflush = module.get_function("fflush").unwrap();
        b.build_call(fflush, &[fp.into()], "").unwrap();
        b.build_unconditional_branch(end_bb).unwrap();

        b.position_at_end(end_bb);
        b.build_return(None).unwrap();
    }

    // ── bruto_capture_writeln() ─────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[], false);
        let func = module.add_function("bruto_capture_writeln", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);

        let g = module.get_global("_bruto_capture_fp").unwrap();
        let fp = b.build_load(ptr_ty, g.as_pointer_value(), "fp").unwrap().into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();

        let write_bb = context.append_basic_block(func, "write");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, end_bb, write_bb).unwrap();

        b.position_at_end(write_bb);
        let fprintf = module.get_function("fprintf").unwrap();
        let fmt = b.build_global_string_ptr("\n", "cap_nl").unwrap();
        b.build_call(fprintf, &[fp.into(), fmt.as_pointer_value().into()], "").unwrap();
        let fflush = module.get_function("fflush").unwrap();
        b.build_call(fflush, &[fp.into()], "").unwrap();
        b.build_unconditional_branch(end_bb).unwrap();

        b.position_at_end(end_bb);
        b.build_return(None).unwrap();
    }

    // ── bruto_capture_close() ───────────────────────────────
    {
        let fn_ty = void_ty.fn_type(&[], false);
        let func = module.add_function("bruto_capture_close", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);

        let g = module.get_global("_bruto_capture_fp").unwrap();
        let fp = b.build_load(ptr_ty, g.as_pointer_value(), "fp").unwrap().into_pointer_value();
        let is_null = b.build_is_null(fp, "null").unwrap();

        let close_bb = context.append_basic_block(func, "close");
        let end_bb = context.append_basic_block(func, "end");
        b.build_conditional_branch(is_null, end_bb, close_bb).unwrap();

        b.position_at_end(close_bb);
        let fclose = module.get_function("fclose").unwrap();
        b.build_call(fclose, &[fp.into()], "").unwrap();
        b.build_store(g.as_pointer_value(), ptr_ty.const_null()).unwrap();
        b.build_unconditional_branch(end_bb).unwrap();

        b.position_at_end(end_bb);
        b.build_return(None).unwrap();
    }

    // ── String operations ───────────────────────────────

    // libc string externs
    let strlen_ty = i64_ty.fn_type(&[ptr_ty.into()], false);
    module.add_function("strlen", strlen_ty, None);

    let strcpy_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
    module.add_function("strcpy", strcpy_ty, None);

    let strcat_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
    module.add_function("strcat", strcat_ty, None);

    let strcmp_ty = i32_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
    module.add_function("strcmp", strcmp_ty, None);

    // bruto_str_concat(a: ptr, b: ptr) -> ptr  (new heap string = a + b)
    {
        let fn_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
        let func = module.add_function("bruto_str_concat", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let a = func.get_nth_param(0).unwrap().into_pointer_value();
        let bv = func.get_nth_param(1).unwrap().into_pointer_value();
        let strlen = module.get_function("strlen").unwrap();
        let len_a = b.build_call(strlen, &[a.into()], "la").unwrap()
            .try_as_basic_value().basic().unwrap().into_int_value();
        let len_b = b.build_call(strlen, &[bv.into()], "lb").unwrap()
            .try_as_basic_value().basic().unwrap().into_int_value();
        let total = b.build_int_add(len_a, len_b, "tot").unwrap();
        let total_plus1 = b.build_int_add(total, i64_ty.const_int(1, false), "tot1").unwrap();
        let malloc = module.get_function("malloc").unwrap();
        let buf = b.build_call(malloc, &[total_plus1.into()], "buf").unwrap()
            .try_as_basic_value().basic().unwrap();
        let strcpy = module.get_function("strcpy").unwrap();
        b.build_call(strcpy, &[buf.into(), a.into()], "").unwrap();
        let strcat = module.get_function("strcat").unwrap();
        b.build_call(strcat, &[buf.into(), bv.into()], "").unwrap();
        b.build_return(Some(&buf)).unwrap();
    }

    // bruto_str_length(s: ptr) -> i64
    {
        let fn_ty = i64_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_str_length", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let strlen = module.get_function("strlen").unwrap();
        let len = b.build_call(strlen, &[s.into()], "len").unwrap()
            .try_as_basic_value().basic().unwrap();
        b.build_return(Some(&len)).unwrap();
    }

    // bruto_str_compare(a: ptr, b: ptr) -> i32  (strcmp wrapper)
    {
        let fn_ty = i32_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
        let func = module.add_function("bruto_str_compare", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let a = func.get_nth_param(0).unwrap().into_pointer_value();
        let bv = func.get_nth_param(1).unwrap().into_pointer_value();
        let strcmp = module.get_function("strcmp").unwrap();
        let r = b.build_call(strcmp, &[a.into(), bv.into()], "cmp").unwrap()
            .try_as_basic_value().basic().unwrap();
        b.build_return(Some(&r)).unwrap();
    }

    // ── libc externs for string ops ────────────────────────
    let memcpy_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into(), i64_ty.into()], false);
    module.add_function("memcpy", memcpy_ty, None);

    let snprintf_ty = i32_ty.fn_type(&[ptr_ty.into(), i64_ty.into(), ptr_ty.into()], true);
    module.add_function("snprintf", snprintf_ty, None);

    let strstr_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
    module.add_function("strstr", strstr_ty, None);

    let atoll_ty = i64_ty.fn_type(&[ptr_ty.into()], false);
    module.add_function("atoll", atoll_ty, None);

    // bruto_str_copy(s: ptr, index: i64, count: i64) -> ptr
    // copy(s, index, count) — 1-based index
    {
        let fn_ty = ptr_ty.fn_type(&[ptr_ty.into(), i64_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_str_copy", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_nth_param(0).unwrap().into_pointer_value();
        let index = func.get_nth_param(1).unwrap().into_int_value();
        let count = func.get_nth_param(2).unwrap().into_int_value();
        // start = index - 1
        let start = b.build_int_sub(index, i64_ty.const_int(1, false), "start").unwrap();
        // allocate count+1 bytes
        let buf_size = b.build_int_add(count, i64_ty.const_int(1, false), "bsz").unwrap();
        let malloc = module.get_function("malloc").unwrap();
        let buf = b.build_call(malloc, &[buf_size.into()], "buf").unwrap()
            .try_as_basic_value().basic().unwrap().into_pointer_value();
        // src = s + start
        let src = unsafe { b.build_in_bounds_gep(context.i8_type(), s, &[start], "src").unwrap() };
        let memcpy = module.get_function("memcpy").unwrap();
        b.build_call(memcpy, &[buf.into(), src.into(), count.into()], "").unwrap();
        // null-terminate: buf[count] = 0
        let end = unsafe { b.build_in_bounds_gep(context.i8_type(), buf, &[count], "end").unwrap() };
        b.build_store(end, context.i8_type().const_int(0, false)).unwrap();
        b.build_return(Some(&buf)).unwrap();
    }

    // bruto_str_pos(substr: ptr, s: ptr) -> i64
    // pos(substr, s) — returns 1-based position, 0 if not found
    {
        let fn_ty = i64_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], false);
        let func = module.add_function("bruto_str_pos", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let substr = func.get_nth_param(0).unwrap().into_pointer_value();
        let s = func.get_nth_param(1).unwrap().into_pointer_value();
        let strstr = module.get_function("strstr").unwrap();
        let found = b.build_call(strstr, &[s.into(), substr.into()], "found").unwrap()
            .try_as_basic_value().basic().unwrap().into_pointer_value();
        let is_null = b.build_is_null(found, "null").unwrap();
        let found_bb = context.append_basic_block(func, "found");
        let not_found_bb = context.append_basic_block(func, "not_found");
        b.build_conditional_branch(is_null, not_found_bb, found_bb).unwrap();

        b.position_at_end(found_bb);
        let s_int = b.build_ptr_to_int(s, i64_ty, "si").unwrap();
        let f_int = b.build_ptr_to_int(found, i64_ty, "fi").unwrap();
        let diff = b.build_int_sub(f_int, s_int, "diff").unwrap();
        let pos = b.build_int_add(diff, i64_ty.const_int(1, false), "pos").unwrap(); // 1-based
        b.build_return(Some(&pos)).unwrap();

        b.position_at_end(not_found_bb);
        b.build_return(Some(&i64_ty.const_int(0, false))).unwrap();
    }

    // bruto_str_delete(s: ptr, index: i64, count: i64) -> ptr
    {
        let fn_ty = ptr_ty.fn_type(&[ptr_ty.into(), i64_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_str_delete", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_nth_param(0).unwrap().into_pointer_value();
        let index = func.get_nth_param(1).unwrap().into_int_value();
        let count = func.get_nth_param(2).unwrap().into_int_value();
        let strlen = module.get_function("strlen").unwrap();
        let slen = b.build_call(strlen, &[s.into()], "slen").unwrap()
            .try_as_basic_value().basic().unwrap().into_int_value();
        // start = index - 1; tail_start = start + count; tail_len = slen - tail_start
        let start = b.build_int_sub(index, i64_ty.const_int(1, false), "start").unwrap();
        let tail_start = b.build_int_add(start, count, "ts").unwrap();
        let tail_len = b.build_int_sub(slen, tail_start, "tl").unwrap();
        // new_len = start + tail_len
        let new_len = b.build_int_add(start, tail_len, "nl").unwrap();
        let buf_size = b.build_int_add(new_len, i64_ty.const_int(1, false), "bsz").unwrap();
        let malloc = module.get_function("malloc").unwrap();
        let buf = b.build_call(malloc, &[buf_size.into()], "buf").unwrap()
            .try_as_basic_value().basic().unwrap().into_pointer_value();
        // copy first part
        let memcpy = module.get_function("memcpy").unwrap();
        b.build_call(memcpy, &[buf.into(), s.into(), start.into()], "").unwrap();
        // copy tail
        let buf_tail = unsafe { b.build_in_bounds_gep(context.i8_type(), buf, &[start], "bt").unwrap() };
        let s_tail = unsafe { b.build_in_bounds_gep(context.i8_type(), s, &[tail_start], "st").unwrap() };
        let tail_plus1 = b.build_int_add(tail_len, i64_ty.const_int(1, false), "tp1").unwrap(); // include null
        b.build_call(memcpy, &[buf_tail.into(), s_tail.into(), tail_plus1.into()], "").unwrap();
        b.build_return(Some(&buf)).unwrap();
    }

    // bruto_str_insert(source: ptr, s: ptr, index: i64) -> ptr
    {
        let fn_ty = ptr_ty.fn_type(&[ptr_ty.into(), ptr_ty.into(), i64_ty.into()], false);
        let func = module.add_function("bruto_str_insert", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let source = func.get_nth_param(0).unwrap().into_pointer_value();
        let s = func.get_nth_param(1).unwrap().into_pointer_value();
        let index = func.get_nth_param(2).unwrap().into_int_value();
        let strlen = module.get_function("strlen").unwrap();
        let slen = b.build_call(strlen, &[s.into()], "slen").unwrap()
            .try_as_basic_value().basic().unwrap().into_int_value();
        let src_len = b.build_call(strlen, &[source.into()], "srclen").unwrap()
            .try_as_basic_value().basic().unwrap().into_int_value();
        let total = b.build_int_add(slen, src_len, "tot").unwrap();
        let buf_size = b.build_int_add(total, i64_ty.const_int(1, false), "bsz").unwrap();
        let malloc = module.get_function("malloc").unwrap();
        let buf = b.build_call(malloc, &[buf_size.into()], "buf").unwrap()
            .try_as_basic_value().basic().unwrap().into_pointer_value();
        let pos = b.build_int_sub(index, i64_ty.const_int(1, false), "pos").unwrap();
        // copy s[0..pos]
        let memcpy = module.get_function("memcpy").unwrap();
        b.build_call(memcpy, &[buf.into(), s.into(), pos.into()], "").unwrap();
        // copy source
        let buf_mid = unsafe { b.build_in_bounds_gep(context.i8_type(), buf, &[pos], "bm").unwrap() };
        b.build_call(memcpy, &[buf_mid.into(), source.into(), src_len.into()], "").unwrap();
        // copy s[pos..] including null
        let buf_tail = unsafe {
            let off = b.build_int_add(pos, src_len, "off").unwrap();
            b.build_in_bounds_gep(context.i8_type(), buf, &[off], "btail").unwrap()
        };
        let s_tail = unsafe { b.build_in_bounds_gep(context.i8_type(), s, &[pos], "stail").unwrap() };
        let tail_len = b.build_int_sub(slen, pos, "tlen").unwrap();
        let tail_plus1 = b.build_int_add(tail_len, i64_ty.const_int(1, false), "tp1").unwrap();
        b.build_call(memcpy, &[buf_tail.into(), s_tail.into(), tail_plus1.into()], "").unwrap();
        b.build_return(Some(&buf)).unwrap();
    }

    // bruto_int_to_str(x: i64) -> ptr
    {
        let fn_ty = ptr_ty.fn_type(&[i64_ty.into()], false);
        let func = module.add_function("bruto_int_to_str", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let x = func.get_first_param().unwrap().into_int_value();
        let malloc = module.get_function("malloc").unwrap();
        let buf = b.build_call(malloc, &[i64_ty.const_int(32, false).into()], "buf").unwrap()
            .try_as_basic_value().basic().unwrap().into_pointer_value();
        let snprintf = module.get_function("snprintf").unwrap();
        let fmt = b.build_global_string_ptr("%lld", "itoa_fmt").unwrap();
        b.build_call(snprintf, &[buf.into(), i64_ty.const_int(32, false).into(), fmt.as_pointer_value().into(), x.into()], "").unwrap();
        b.build_return(Some(&buf)).unwrap();
    }

    // bruto_real_to_str(x: f64) -> ptr
    {
        let f64_ty = context.f64_type();
        let fn_ty = ptr_ty.fn_type(&[f64_ty.into()], false);
        let func = module.add_function("bruto_real_to_str", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let x = func.get_first_param().unwrap().into_float_value();
        let malloc = module.get_function("malloc").unwrap();
        let buf = b.build_call(malloc, &[i64_ty.const_int(64, false).into()], "buf").unwrap()
            .try_as_basic_value().basic().unwrap().into_pointer_value();
        let snprintf = module.get_function("snprintf").unwrap();
        let fmt = b.build_global_string_ptr("%.10g", "ftoa_fmt").unwrap();
        b.build_call(snprintf, &[buf.into(), i64_ty.const_int(64, false).into(), fmt.as_pointer_value().into(), x.into()], "").unwrap();
        b.build_return(Some(&buf)).unwrap();
    }

    // bruto_str_to_int(s: ptr) -> i64
    {
        let fn_ty = i64_ty.fn_type(&[ptr_ty.into()], false);
        let func = module.add_function("bruto_str_to_int", fn_ty, None);
        let bb = context.append_basic_block(func, "entry");
        let b = context.create_builder();
        b.position_at_end(bb);
        let s = func.get_first_param().unwrap().into_pointer_value();
        let atoll = module.get_function("atoll").unwrap();
        let r = b.build_call(atoll, &[s.into()], "val").unwrap()
            .try_as_basic_value().basic().unwrap();
        b.build_return(Some(&r)).unwrap();
    }
}

/// Helper: emit a `bruto_capture_write_*` function that fprintf's a single
/// value through a format string to the capture file (if open).
fn emit_capture_write_fn<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    name: &str,
    param_types: &[inkwell::types::BasicMetadataTypeEnum<'ctx>],
    fmt_str: &str,
) {
    let void_ty = context.void_type();
    let ptr_ty = context.ptr_type(AddressSpace::default());
    let fn_ty = void_ty.fn_type(param_types, false);
    let func = module.add_function(name, fn_ty, None);
    let bb = context.append_basic_block(func, "entry");
    let b = context.create_builder();
    b.position_at_end(bb);

    let g = module.get_global("_bruto_capture_fp").unwrap();
    let fp = b.build_load(ptr_ty, g.as_pointer_value(), "fp").unwrap().into_pointer_value();
    let is_null = b.build_is_null(fp, "null").unwrap();

    let write_bb = context.append_basic_block(func, "write");
    let end_bb = context.append_basic_block(func, "end");
    b.build_conditional_branch(is_null, end_bb, write_bb).unwrap();

    b.position_at_end(write_bb);
    let fprintf = module.get_function("fprintf").unwrap();
    let fmt = b.build_global_string_ptr(fmt_str, "cap_fmt").unwrap();
    let val = func.get_first_param().unwrap();
    b.build_call(fprintf, &[fp.into(), fmt.as_pointer_value().into(), val.into()], "").unwrap();
    let fflush = module.get_function("fflush").unwrap();
    b.build_call(fflush, &[fp.into()], "").unwrap();
    b.build_unconditional_branch(end_bb).unwrap();

    b.position_at_end(end_bb);
    b.build_return(None).unwrap();
}
