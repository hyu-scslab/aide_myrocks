# Copyright (c) 2006, 2016, Oracle and/or its affiliates. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; version 2 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA

INCLUDE_DIRECTORIES(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/sql
  ${CMAKE_SOURCE_DIR}/utils
  ${CMAKE_SOURCE_DIR}/regex
  ${ZLIB_INCLUDE_DIR}
  ${SSL_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/sql
)

IF(WITH_JUNCTION)
  SET(JUNCTION_LIBRARIES ${JUNCTION_LIBRARIES}
      ${WITH_JUNCTION}/lib/libjunction${PIC_EXT}.a)
  SET(JUNCTION_LIBRARIES ${JUNCTION_LIBRARIES}
    ${WITH_TURF}/lib/libturf${PIC_EXT}.a)
  ADD_DEFINITIONS(-DHAVE_JUNCTION)
ENDIF()

IF(WITH_RAPIDJSON)
  ADD_DEFINITIONS(-DHAVE_RAPIDJSON)
ENDIF()

SET(GEN_SOURCES
  ${CMAKE_CURRENT_BINARY_DIR}/sql_yacc.h
  ${CMAKE_CURRENT_BINARY_DIR}/sql_yacc.cc
  ${CMAKE_CURRENT_BINARY_DIR}/sql_builtin.cc
  ${CMAKE_CURRENT_BINARY_DIR}/lex_hash.h
  ${CMAKE_CURRENT_BINARY_DIR}/mysql_githash.h
)
SET(GEN_DIGEST_SOURCES
  ${CMAKE_CURRENT_BINARY_DIR}/lex_token.h
)

SET_SOURCE_FILES_PROPERTIES(${GEN_SOURCES}
                            ${GEN_DIGEST_SOURCES}
                            PROPERTIES GENERATED 1)

# Gen_lex_token
# Make sure sql_yacc.h is generated before compiling gen_lex_token
ADD_EXECUTABLE(gen_lex_token gen_lex_token.cc)
ADD_DEPENDENCIES(gen_lex_token GenServerSource)

ADD_CUSTOM_COMMAND(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lex_token.h
  COMMAND gen_lex_token > lex_token.h
  DEPENDS gen_lex_token
)

ADD_DEFINITIONS(-DMYSQL_SERVER -DHAVE_EVENT_SCHEDULER)
IF(SSL_DEFINES)
 ADD_DEFINITIONS(${SSL_DEFINES})
ENDIF()

SET(SQL_SHARED_SOURCES
  abstract_query_plan.cc
  column_statistics.cc
  datadict.cc
  debug_sync.cc
  db_stats.cc
  derror.cc
  des_key_file.cc
  discover.cc
  field.cc
  field_conv.cc
  filesort.cc
  filesort_utils.cc
  gcalc_slicescan.cc
  gcalc_tools.cc
  slave_stats_daemon.cc
  gstream.cc
  handler.cc
  hh_wheel_timer.cc
  hostname.cc
  hyperloglog.cc
  init.cc
  item.cc
  item_buff.cc
  item_cmpfunc.cc
  item_create.cc
  item_document.cc
  item_func.cc
  item_geofunc.cc
  item_row.cc
  item_strfunc.cc
  item_subselect.cc
  item_sum.cc
  item_sum_hll.cc
  item_timefunc.cc
  item_xmlfunc.cc
  item_inetfunc.cc
  item_jsonfunc.cc
  key.cc
  keycaches.cc
  lock.cc
  log.cc
  mdl.cc
  mf_iocache.cc
  my_decimal.cc
  native_procedure.cc
  net_serv.cc
  opt_explain.cc
  opt_explain_traditional.cc
  opt_explain_json.cc
  opt_range.cc
  opt_sum.cc
  opt_trace.cc
  opt_trace2server.cc
  parse_file.cc
  partition_info.cc
  password.c
  procedure.cc
  protocol.cc
  query_tag_perf_counter.cc
  records.cc
  rpl_handler.cc
  scheduler.cc
  session_tracker.cc
  set_var.cc
  signal_handler.cc
  sp.cc
  sp_cache.cc
  sp_head.cc
  sp_instr.cc
  sp_pcontext.cc
  sp_rcontext.cc
  spatial.cc
  string_service.cc
  sql_acl.cc
  sql_admin.cc
  sql_alloc_error_handler.cc
  sql_alter.cc
  sql_analyse.cc
  sql_audit.cc
  sql_base.cc
  sql_bootstrap.cc
  sql_cache.cc
  sql_class.cc
  shardedlocks.cc
  sql_connect.cc
  sql_crypt.cc
  sql_cursor.cc
  sql_data_change.cc
  sql_db.cc
  sql_delete.cc
  sql_derived.cc
  sql_digest.cc
  sql_do.cc
  sql_error.cc
  sql_executor.cc
  sql_get_diagnostics.cc
  sql_handler.cc
  sql_help.cc
  sql_insert.cc
  sql_join_buffer.cc
  sql_lex.cc
  sql_list.cc
  sql_load.cc
  sql_locale.cc
  sql_manager.cc
  sql_multi_tenancy.cc
  sql_optimizer.cc
  sql_parse.cc
  sql_parse_com_rpc.cc
  sql_partition.cc
  sql_partition_admin.cc
  sql_planner.cc
  sql_plans.cc
  sql_plugin.cc
  sql_prepare.cc
  sql_profile.cc
  sql_reload.cc
  sql_rename.cc
  sql_resolver.cc
  sql_rewrite.cc
  sql_select.cc
  sql_servers.cc
  srv_session.cc
  sql_show.cc
  sql_signal.cc
  sql_state.c
  sql_stats.cc
  sql_string.cc
  sql_table.cc
  sql_tablespace.cc
  sql_test.cc
  sql_time.cc
  sql_tmp_table.cc
  sql_trigger.cc
  sql_truncate.cc
  sql_udf.cc
  sql_union.cc
  sql_update.cc
  sql_view.cc
  strfunc.cc
  sys_vars.cc
  table.cc
  table_cache.cc
  thr_malloc.cc
  timer.cc
  transaction.cc
  tztime.cc
  uniques.cc
  unireg.cc
)

SET(SQL_EXPORTED_SOURCES ${SQL_SHARED_SOURCES} PARENT_SCOPE)

SET(SQL_SOURCE
  ${GEN_SOURCES}
  ${GEN_DIGEST_SOURCES}
  ${MYSYS_LIBWRAP_SOURCE}
  ${SQL_SHARED_SOURCES}
  ../libmysql/errmsg.c
  ../sql-common/client.c
  ../sql-common/client_plugin.c
  ../sql-common/my_time.c
  ../sql-common/my_user.c
  ../sql-common/pack.c
  ../sql-common/client_authentication.cc
  event_data_objects.cc
  event_db_repository.cc
  event_parse_data.cc
  event_queue.cc
  event_scheduler.cc
  events.cc
  mysqld.cc
  sql_client.cc
  table_stats.cc
  error_stats.cc
  )

IF(WIN32)
  LIST(APPEND SQL_SOURCE named_pipe.cc)
ENDIF()

IF (WIN32 AND OPENSSL_APPLINK_C)
  SET_SOURCE_FILES_PROPERTIES(
    ../sql-common/client_authentication.cc
    PROPERTIES COMPILE_FLAGS "-DHAVE_OPENSSL_APPLINK_C"
  )
ENDIF()

IF(HAVE_MY_TIMER)
  SET(SQL_SOURCE ${SQL_SOURCE} sql_timer.cc)
ENDIF()

MYSQL_ADD_PLUGIN(partition ha_partition.cc STORAGE_ENGINE DEFAULT STATIC_ONLY
RECOMPILE_FOR_EMBEDDED)

IF(WITH_NDP_LIB)  
	FIND_LIBRARY(SMARTSSD_LIBRARY NAMES ndpf HINTS /opt/xilinx/xrt/lib)
	MESSAGE(STATUS "SMARTSSD_LIBRARY= ${SMARTSSD_LIBRARY}")

	FIND_LIBRARY(XILINX_LIBRARY NAMES xilinxopencl HINTS /opt/xilinx/xrt/lib)
	MESSAGE(STATUS "XILINX_LIBRARY= ${XILINX_LIBRARY}")
ENDIF()

ADD_LIBRARY(sql STATIC ${SQL_SOURCE})
ADD_DEPENDENCIES(sql GenServerSource)
ADD_DEPENDENCIES(sql GenDigestServerSource)
DTRACE_INSTRUMENT(sql)
TARGET_LINK_LIBRARIES(sql ${MYSQLD_STATIC_PLUGIN_LIBS}
  mysys mysys_ssl dbug strings vio regex utils)
IF(WITH_NDP_LIB)  
  TARGET_LINK_LIBRARIES(sql ${SMARTSSD_LIBRARY} ${XILINX_LIBRARY})
ENDIF()
TARGET_LINK_LIBRARIES(sql ${LIBWRAP} ${LIBCRYPT} ${LIBDL}
  ${SSL_LIBRARIES} ${LIBCAP})
#
# On Windows platform we compile in the clinet-side Windows Native Authentication
# plugin which is used by the client connection code included in the server.
#
IF(WIN32)
  ADD_DEFINITIONS(-DAUTHENTICATION_WIN)
  TARGET_LINK_LIBRARIES(sql auth_win_client)
ENDIF()

SET (BINLOG_SOURCE uuid.cc rpl_gtid_misc.cc
                   rpl_gtid_sid_map.cc rpl_gtid_set.cc rpl_gtid_specification.cc
                   rpl_gtid_state.cc rpl_gtid_owned.cc rpl_gtid_cache.cc
                   rpl_gtid_execution.cc rpl_gtid_mutex_cond_array.cc
                   log_event.cc log_event_old.cc binlog.cc sql_binlog.cc
       rpl_filter.cc rpl_record.cc rpl_record_old.cc rpl_utility.cc
       rpl_injector.cc log_event_wrapper.cc)
ADD_LIBRARY(binlog ${BINLOG_SOURCE})
SET (RPL_SOURCE rpl_handler.cc rpl_tblmap.cc)
ADD_DEPENDENCIES(binlog GenError)
ADD_LIBRARY(rpl ${RPL_SOURCE})
SET (MASTER_SOURCE rpl_master.cc)
ADD_DEPENDENCIES(rpl GenError)
ADD_LIBRARY(master ${MASTER_SOURCE})
ADD_DEPENDENCIES(master GenError)
TARGET_LINK_LIBRARIES(master ${JUNCTION_LIBRARIES})
SET (SLAVE_SOURCE rpl_slave.cc rpl_reporting.cc rpl_mi.cc rpl_rli.cc
      rpl_info_handler.cc rpl_info_file.cc rpl_info_table.cc
      rpl_info_values.cc rpl_info.cc rpl_info_factory.cc
      rpl_info_table_access.cc dynamic_ids.cc rpl_rli_pdb.cc
      rpl_slave_commit_order_manager.cc
      rpl_gtid_info.cc rpl_info_dummy.cc dependency_slave_worker.cc)
ADD_LIBRARY(slave ${SLAVE_SOURCE})
ADD_DEPENDENCIES(slave GenError)
ADD_LIBRARY(sqlgunitlib
  filesort_utils.cc mdl.cc sql_list.cc sql_string.cc thr_malloc.cc
  )
ADD_DEPENDENCIES(sqlgunitlib GenError)


IF(WIN32)
  SET(MYSQLD_SOURCE main.cc nt_servc.cc nt_servc.h message.rc)
ELSE()
  SET(MYSQLD_SOURCE main.cc ${DTRACE_PROBES_ALL})
ENDIF()

IF(HAVE_BUILD_INFO)
  SET(MYSQLD_SOURCE ${MYSQLD_SOURCE} ${CMAKE_CURRENT_BINARY_DIR}/mysqld.build_info.c)
ENDIF()

MYSQL_ADD_EXECUTABLE(mysqld ${MYSQLD_SOURCE} DESTINATION ${INSTALL_SBINDIR} COMPONENT Server)

OPTION(DEBUG_EXTNAME "Build server as mysqld-debug (debug builds only)" OFF)
MARK_AS_ADVANCED(DEBUG_EXTNAME)

IF(DEBUG_EXTNAME)
  SET_TARGET_PROPERTIES(mysqld PROPERTIES DEBUG_OUTPUT_NAME "mysqld-debug")
ENDIF()

IF(APPLE)
  # Add CoreServices framework since some dloadable plugins may need it
  FIND_LIBRARY(CORESERVICES NAMES CoreServices)
  IF(CORESERVICES)
    TARGET_LINK_LIBRARIES(mysqld ${CORESERVICES})
  ENDIF()
ENDIF()

IF(NOT WITHOUT_DYNAMIC_PLUGINS)
  SET_TARGET_PROPERTIES(mysqld PROPERTIES ENABLE_EXPORTS TRUE)
  GET_TARGET_PROPERTY(mysqld_link_flags mysqld LINK_FLAGS)
  IF(NOT mysqld_link_flags)
    SET(mysqld_link_flags)
  ENDIF()
  IF (MINGW OR CYGWIN)
    SET_TARGET_PROPERTIES(mysqld PROPERTIES LINK_FLAGS "${mysqld_link_flags} -Wl,--export-all-symbols")
  ENDIF()
  IF(MSVC)
    # Set module definition file. Also use non-incremental linker,
    # incremental appears to crash from time to time,if used with /DEF option
    SET_TARGET_PROPERTIES(mysqld PROPERTIES LINK_FLAGS "${mysqld_link_flags} /DEF:mysqld.def /INCREMENTAL:NO")

    FOREACH (CORELIB sql mysys mysys_ssl dbug strings)
      GET_TARGET_PROPERTY(LOC ${CORELIB} LOCATION)
      FILE(TO_NATIVE_PATH ${LOC} LOC)
      SET (LIB_LOCATIONS ${LIB_LOCATIONS} ${LOC})
    ENDFOREACH (CORELIB ${MYSQLD_CORE_LIBS})
    SET(_PLATFORM x86)
    IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
      SET(_PLATFORM  x64)
    ENDIF()
    ADD_CUSTOM_COMMAND(TARGET mysqld PRE_LINK
      COMMAND echo ${_PLATFORM} && cscript ARGS //nologo ${PROJECT_SOURCE_DIR}/win/create_def_file.js
                  ${_PLATFORM}  ${LIB_LOCATIONS} > mysqld.def
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    ADD_DEPENDENCIES(sql GenError)
  ENDIF()
ENDIF()

SET_TARGET_PROPERTIES(mysqld PROPERTIES ENABLE_EXPORTS TRUE)
TARGET_LINK_LIBRARIES(mysqld sql)
IF(WITH_NDP_LIB)  
  TARGET_LINK_LIBRARIES(sql ${SMARTSSD_LIBRARY})
ENDIF()
TARGET_LINK_LIBRARIES(mysqld binlog rpl master slave sql mysys mysys_ssl)

# Provide plugins with minimal set of libraries
SET(INTERFACE_LIBS ${LIBRT})
IF(INTERFACE_LIBS)
 SET_TARGET_PROPERTIES(mysqld PROPERTIES LINK_INTERFACE_LIBRARIES
  "${INTERFACE_LIBS}")
ENDIF()

# On Solaris, some extra effort is required in order to get dtrace probes
# from static libraries
DTRACE_INSTRUMENT_STATIC_LIBS(mysqld
 "sql;mysys;mysys_ssl;${MYSQLD_STATIC_PLUGIN_LIBS}")


SET(WITH_MYSQLD_LDFLAGS "" CACHE STRING "Additional linker flags for mysqld")
MARK_AS_ADVANCED(WITH_MYSQLD_LDFLAGS)
IF(WITH_MYSQLD_LDFLAGS)
  GET_TARGET_PROPERTY(mysqld LINK_FLAGS MYSQLD_LINK_FLAGS)
  IF(NOT MYSQLD_LINK_FLAGS)
    SET(MYSQLD_LINK_FLAGS)
  ENDIF()
  SET_TARGET_PROPERTIES(mysqld PROPERTIES LINK_FLAGS
     "${MYSQLD_LINK_FLAGS} ${WITH_MYSQLD_LDFLAGS}")
ENDIF()

IF(DEBUG_EXTNAME)
  INSTALL_DEBUG_TARGET(mysqld
    DESTINATION ${INSTALL_SBINDIR}
    PDB_DESTINATION ${INSTALL_SBINDIR}
    SUFFIX "-debug")
ELSE()
  INSTALL_DEBUG_TARGET(mysqld
    DESTINATION ${INSTALL_SBINDIR}
    PDB_DESTINATION ${INSTALL_SBINDIR}/debug
    RENAME mysqld-debug)
ENDIF()

# Handle out-of-source build from source package with possibly broken
# bison. Copy bison output to from source to build directory, if not already
# there
IF (NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
  IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/sql_yacc.cc)
    IF(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/sql_yacc.cc)
      CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/sql_yacc.cc
        ${CMAKE_CURRENT_BINARY_DIR}/sql_yacc.cc COPYONLY)
      CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/sql_yacc.h
        ${CMAKE_CURRENT_BINARY_DIR}/sql_yacc.h COPYONLY)
    ENDIF()
  ENDIF()
ENDIF()


INCLUDE(${CMAKE_SOURCE_DIR}/cmake/bison.cmake)
RUN_BISON(
  ${CMAKE_CURRENT_SOURCE_DIR}/sql_yacc.yy
  ${CMAKE_CURRENT_BINARY_DIR}/sql_yacc.cc
  ${CMAKE_CURRENT_BINARY_DIR}/sql_yacc.h
)

# Gen_lex_hash
ADD_EXECUTABLE(gen_lex_hash gen_lex_hash.cc)

ADD_CUSTOM_COMMAND(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lex_hash.h
  COMMAND gen_lex_hash > lex_hash.h
  DEPENDS gen_lex_hash
)

MYSQL_ADD_EXECUTABLE(mysql_tzinfo_to_sql tztime.cc)
SET_TARGET_PROPERTIES(mysql_tzinfo_to_sql PROPERTIES COMPILE_FLAGS "-DTZINFO2SQL")
TARGET_LINK_LIBRARIES(mysql_tzinfo_to_sql mysys mysys_ssl)

ADD_CUSTOM_TARGET(
        GenServerSource
        DEPENDS ${GEN_SOURCES}
)

ADD_CUSTOM_TARGET(
        GenDigestServerSource
        DEPENDS ${GEN_DIGEST_SOURCES}
)

#Need this only for embedded
SET_TARGET_PROPERTIES(GenServerSource PROPERTIES EXCLUDE_FROM_ALL TRUE)

IF(WIN32 OR HAVE_DLOPEN AND NOT DISABLE_SHARED)
  ADD_LIBRARY(udf_example MODULE udf_example.cc)
  SET_TARGET_PROPERTIES(udf_example PROPERTIES PREFIX "")
  # udf_example depends on strings
  IF(WIN32)
    IF(MSVC)
     SET_TARGET_PROPERTIES(udf_example PROPERTIES LINK_FLAGS "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/udf_example.def")
    ENDIF()
    TARGET_LINK_LIBRARIES(udf_example strings)
  ELSE()
    # udf_example is using safemutex exported by mysqld
    TARGET_LINK_LIBRARIES(udf_example mysqld)
    SET_TARGET_PROPERTIES(udf_example
      PROPERTIES LINK_FLAGS "${CMAKE_SHARED_LIBRARY_C_FLAGS}")
  ENDIF()
ENDIF()

IF(NOT WIN32 AND HAVE_DLOPEN AND NOT DISABLE_SHARED)
  ADD_LIBRARY(np_example MODULE np_example.cc)
  SET_TARGET_PROPERTIES(np_example PROPERTIES PREFIX "")
  SET_TARGET_PROPERTIES(np_example
    PROPERTIES LINK_FLAGS "${CMAKE_SHARED_LIBRARY_C_FLAGS}")
ENDIF()

FOREACH(tool glibtoolize libtoolize aclocal autoconf autoheader automake gtar
 tar git)
 STRING(TOUPPER ${tool}  TOOL)
 FIND_PROGRAM(${TOOL}_EXECUTABLE ${tool} DOC "path to the executable")
 MARK_AS_ADVANCED(${TOOL}_EXECUTABLE)
ENDFOREACH()

CONFIGURE_FILE(
  ${CMAKE_SOURCE_DIR}/cmake/make_dist.cmake.in
  ${CMAKE_BINARY_DIR}/make_dist.cmake @ONLY)

ADD_CUSTOM_TARGET(dist
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/make_dist.cmake
  DEPENDS ${CMAKE_BINARY_DIR}/sql/sql_yacc.cc ${CMAKE_BINARY_DIR}/sql/sql_yacc.h
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

ADD_CUSTOM_TARGET(distclean
  COMMAND ${CMAKE_COMMAND} -E echo  WARNING: distclean target is not functional
  COMMAND ${CMAKE_COMMAND} -E echo  Use 'git clean -fdx' instead
  VERBATIM
  )

IF(INSTALL_LAYOUT STREQUAL "STANDALONE")

# Copy db.opt into data/test/
SET(DBOPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/db.opt )
INSTALL(FILES ${DBOPT_FILE} DESTINATION data/test COMPONENT DataFiles)

# Install initial database on windows
IF(NOT CMAKE_CROSSCOMPILING)
  IF(DEBUG_EXTNAME)
    GET_TARGET_PROPERTY(MYSQLD_EXECUTABLE mysqld LOCATION_DEBUG)
  ELSE()
    GET_TARGET_PROPERTY(MYSQLD_EXECUTABLE mysqld LOCATION)
  ENDIF()
ENDIF()
IF(WIN32 AND MYSQLD_EXECUTABLE)
  CONFIGURE_FILE(
    ${CMAKE_SOURCE_DIR}/cmake/create_initial_db.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/create_initial_db.cmake
    @ONLY
  )

  IF(MSVC_IDE OR CMAKE_GENERATOR MATCHES "Xcode")
    SET (CONFIG_PARAM -DCONFIG=${CMAKE_CFG_INTDIR})
  ENDIF()
  MAKE_DIRECTORY(${CMAKE_CURRENT_BINARY_DIR}/data)
  ADD_CUSTOM_COMMAND(
     OUTPUT initdb.dep
     COMMAND ${CMAKE_COMMAND}
     ${CONFIG_PARAM} -P ${CMAKE_CURRENT_BINARY_DIR}/create_initial_db.cmake
     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data
     DEPENDS mysqld
  )
  ADD_CUSTOM_TARGET(initial_database
    ALL
    DEPENDS  initdb.dep
  )
  INSTALL(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data
    DESTINATION .
    COMPONENT DataFiles
    PATTERN "initdb.dep" EXCLUDE
    PATTERN "bootstrap.sql" EXCLUDE)
ELSE()
  # Not windows or cross compiling, just install an empty directory
  INSTALL(FILES ${DUMMY_FILE} DESTINATION data/mysql COMPONENT DataFiles)
ENDIF()
ENDIF()

ADD_CUSTOM_TARGET(show-dist-name
  COMMAND ${CMAKE_COMMAND} -E echo "${CPACK_PACKAGE_FILE_NAME}"
)

# If the repository was modified (via push or pull) or if the
# mysql_githash.h.in changed rebuild the mysql_githash.h file
SET(GITHASH_HEADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/mysql_githash.h.in)
SET(GITHASH_HEADER ${CMAKE_CURRENT_BINARY_DIR}/mysql_githash.h)

SET(GITHASH_DEPENDS ${GITHASH_HEADER_SOURCE})
SET(GITHASH_PERL_OPTS --git_root=${CMAKE_SOURCE_DIR} --file=${GITHASH_HEADER})
IF(MYSQL_GITHASH)
  # If the hashes and dates are passed in add them to the parameters for
  # the perl script
  SET(GITHASH_PERL_OPTS ${GITHASH_PERL_OPTS}
                        --mysql_githash=${MYSQL_GITHASH}
                        --mysql_gitdate=${MYSQL_GITDATE}
                        --rocksdb_githash=${ROCKSDB_GITHASH}
                        --rocksdb_gitdate=${ROCKSDB_GITDATE})
ENDIF()

SET(GITHASH_PERL ${CMAKE_SOURCE_DIR}/cmake/githash.pl)
ADD_CUSTOM_COMMAND(OUTPUT ${GITHASH_HEADER}
  COMMAND ${CMAKE_COMMAND} -E copy ${GITHASH_HEADER_SOURCE} ${GITHASH_HEADER}
  COMMAND ${CMAKE_COMMAND} -E echo ${GITHASH_PERL} ${GITHASH_PERL_OPTS}
  COMMAND perl ${GITHASH_PERL} ${GITHASH_PERL_OPTS}
  DEPENDS ${GITHASH_DEPENDS}
  VERBATIM
)
